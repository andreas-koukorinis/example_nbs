from heapq import heappop, heappush
from datetime import datetime, timedelta
from threading import Condition
from stratagemdataprocessing.util.thread_manager import DaemonThreadExceptionManager
import traceback


# sentinel alarm value
import pytz
from sgmtradingcore.core.ts_log import get_logger

ALARM_TICK = 1


class Clock(object):
    def __init__(self):
        self._sim_time = None

    def set_time(self, time):
        self._sim_time = time

    def wall_clock_time(self):
        return datetime.now(tz=pytz.timezone('Europe/London'))

    def sim_time(self):
        return self._sim_time


class TimeSeriesEngine(object):
    """
    The engine is what makes a backtest or realtime run tick. It is responsible
    for keeping track of time and notifying various units when they are supposed
    to produce ticks.
    """

    def __init__(self):
        """
        Create a simulation engine. The engine should only be used in a
        single run
        """
        self._alarm_queue = []
        self._event_queue = []
        self._event_set = set()
        self._activated_set = set()
        self._clock = Clock()
        self._end_time = None
        self._logger = get_logger(__name__)

        self._stopped = False

    def start(self, start_time, end_time=None):
        """
        Entry point for a run
        """
        raise NotImplementedError('Please implement this method')

    def stop(self):
        self._stopped = True

    def event(self, unit):
        """
        Ask the engine to tick a unit in the current cycle or immediately
        start a cycle if one is not active. This is used to activate
        units that have been called by other units or for asynchronous
        real time events to be allowed to produce a value
        """
        if unit in self._activated_set:
            raise ValueError('Attempted to schedule event for unit already activated in this cycle')

        if unit not in self._event_set:
            heappush(self._event_queue, (unit.rank, unit))
            self._event_set.add(unit)

    def alarm(self, alarm_time, the_input, alarm_value=ALARM_TICK):
        """
        Ask the engine to register an alarm, this is useful for ts_defs that
        need to follow up on some input tick or for timeseries providers
        to wake up in order to provide the next set of ticks
        """
        current_sim_time = self._clock.sim_time()

        if isinstance(alarm_time, timedelta):
            alarm_time = current_sim_time + alarm_time

        # alarms set when units are starting will end up here before the
        # current time of the engine has been initialized but will never
        # be timedeltas so we do a None check here and not above
        if current_sim_time is not None and alarm_time <= current_sim_time:
            if hasattr(alarm_value, '__iter__'):
                value_str = ' | '.join([str(v) for v in alarm_value])
            else:
                value_str = str(alarm_value)
            raise ValueError('Attempted to set alarm in the past ({}) [{}]. current_sim_time is {}'.format(
                alarm_time.isoformat(), value_str, current_sim_time))

        heappush(self._alarm_queue, (alarm_time, the_input, alarm_value))

    def async_alarm(self, the_input, alarm_value=ALARM_TICK):
        raise NotImplementedError('Can only be called on the realtime engine')

    def now(self):
        return self._clock.sim_time()

    def _next_alarm_time(self):
        (next_alarm_time, _input, _alarm_value) = (
            self._alarm_queue[0] if len(self._alarm_queue) > 0
            else (None, None, None))

        return next_alarm_time

    def _pop_alarms(self):
        """
        Pop all alarms at the next available time and tick the related inputs
        returning the time of the alarms
        """
        if len(self._alarm_queue) == 0:
            return None

        # get the time of the next alarm
        # the time of the step will be that of the first alarm popped
        # we might support time resolution later but that comes with some
        # problems in that the the units must cooperate
        (alarm_time, the_input, alarm_value) = heappop(self._alarm_queue)
        the_input.tick(alarm_time, alarm_value)

        # pop all alarms within the time resolution currently 0
        while len(self._alarm_queue) > 0:
            if self._alarm_queue[0][0] == alarm_time:
                (alarm_time, the_input, alarm_value) = heappop(self._alarm_queue)
                the_input.tick(alarm_time, alarm_value)
            else:
                break

        return alarm_time

    def _process_events(self):
        """
        Call when at least one event has been scheduled to
        start processing the current engine cycle
        """
        # tick units which will be in rank order
        while len(self._event_queue):
            unit = heappop(self._event_queue)[1]
            unit.activate()
            self._activated_set.add(unit)

        self._event_set.clear()
        self._activated_set.clear()


class SimulationEngine(TimeSeriesEngine):
    """
    The simulation engine only supports alarms to align data
    sources and fast forwards time as quickly as possible
    """

    def __init__(self):
        TimeSeriesEngine.__init__(self)

    def start(self, start_time, end_time=None):
        """
        Entry point for a run
        """
        self._end_time = end_time

        terminate = False

        while not terminate:
            terminate = self.step()

    def step(self):
        """
        Do an engine step ie process all events at a fixed point in time.
        Some code is repeated but intentionally inlined for performance.
        """

        # get the time of the next alarm
        next_alarm_time = self._pop_alarms()

        # stop if we have an end time and have gone past it or stop was requested
        if ((next_alarm_time is None) or (self._end_time is not None and
                next_alarm_time > self._end_time) or self._stopped):
            return True

        self._clock.set_time(next_alarm_time)

        self._process_events()

        return False


class RealtimeEngine(TimeSeriesEngine):
    """
    The realtime engine supports alarms or external events. If there
    is nothing to do it will simply sleep until the real end time is reached.
    """

    def __init__(self):
        TimeSeriesEngine.__init__(self)
        self._condition = Condition()
        self._thread_manager = DaemonThreadExceptionManager(self._condition)
        self._exceptions_queue = self._thread_manager.get_queue()

    def get_thread_manager(self):
        return self._thread_manager

    def start(self, start_time, end_time=None):
        """
        Entry point for a run
        """
        if end_time is None:
            raise ValueError('Cannot run a realtime engine without an '
                             'end time')

        self._end_time = end_time

        terminate = False
        while not terminate:
            # we will generally do something or sleep. we will sleep until
            # the next alarm or the end time whatever comes first,
            # or until an external event arrives
            wall_clock_time = self._clock.wall_clock_time()
            self._condition.acquire()

            # check the queue to see if it has any exceptions
            if not self._exceptions_queue.empty():
                exc_info = self._exceptions_queue.get_nowait()
                # logging.info(item)
                if exc_info:
                    traceback.print_exception(*exc_info)
                    raise exc_info[1]

            next_alarm_time = self._next_alarm_time()
            next_wake_up_time = (min(next_alarm_time, end_time)
                                 if next_alarm_time is not None else end_time)

            if wall_clock_time < next_wake_up_time:
                # we sleep
                time_diff = next_wake_up_time - wall_clock_time
                self._condition.wait(time_diff.seconds + time_diff.microseconds / 1e6)
            else:
                # we do something
                if (next_wake_up_time is not None) and (next_wake_up_time < end_time) and not self._stopped:
                    # we do a step
                    self.step(next_wake_up_time, True)
                else:
                    # we terminate
                    terminate = True

            self._condition.release()

    def step(self, step_time, pop_alarms):
        if pop_alarms:
            self._pop_alarms()
        self._clock.set_time(step_time)
        self._process_events()

    def async_alarm(self, the_input, alarm_value=ALARM_TICK):
        """
        Create an alarm at the current wall clock time which will be drained
        into the normal alarm queue on the next step
        """
        with self._condition:
            self.alarm(self._clock.wall_clock_time(), the_input, alarm_value)
            self._condition.notify()
