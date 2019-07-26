from collections import defaultdict, deque
import copy
import inspect

from pandas.core.series import Series

from sgmtradingcore.core.engines import ALARM_TICK


# input and output latest value port types
TS_SCALAR = "scalar"
TS_MULTI_SCALAR = "mscalar"
TS_LIST = "list"
TS_DICT = "dict"

# port types
TS_INPUT = "ts_input"
TS_OUTPUT = "ts_output"

# other unit definitions
TS_INIT_ALARM = "ts_init_alarm"


class InputPort(object):
    """
    Timeseries input for a TimeSeriesUnit's action function
    """

    def __init__(self, unit, name):
        self.ticked = False
        self._owner = unit
        self._name = name

    def owner(self):
        return self._owner


class TimeSeriesPort(InputPort):
    """
    Wrapper around the cached values for a timeseries to provide
    convenience functions for checking if the timeseries has just
    ticked and getting latest values, values at a particular point
    in time etc
    """

    def __init__(self, unit, name, history=0):
        InputPort.__init__(self, unit, name)
        self._history = history
        self._value_history = deque(maxlen=history)

    def tick(self, time, value):
        self._do_tick(time, value)
        self._owner.schedule()

    def _do_tick(self, time, value):
        raise NotImplementedError('Please Implement this method')

    def alarm(self, time, value):
        self._owner.alarm(time, self, value)

    def value(self):
        raise NotImplementedError('Please Implement this method')

    def _copy_value(self):
        raise NotImplementedError('Please Implement this method')

    def prev_value(self, index=0):
        return self._value_history[index]

    def empty(self):
        raise NotImplementedError('Please Implement this method')

    def reset(self):
        if self._history > 0 and self.empty() is False:
            self._value_history.appendleft(self._copy_value())
        self._do_reset()

    def _do_reset(self):
        raise NotImplementedError('Please Implement this method')


class ScalarPort(TimeSeriesPort):
    def __init__(self, unit, name, default=None, history=0):
        TimeSeriesPort.__init__(self, unit, name, history=history)
        self._image_time = None
        self._image = None
        self._default = default

    def value(self):
        if self._image is None:
            if self._default is not None:
                return self._default
        else:
            return self._image

        raise ValueError('No tick or default value in port %s', self._name)

    def _copy_value(self):
        return copy.copy(self._image)

    def empty(self):
        return self._image is None and self._default is None

    def _do_tick(self, time, value):
        if value is not None:
            if time == self._image_time:
                raise ValueError('Multiple ticks at the same time for port %s', self._name)
            else:
                self._image = value

            self._image_time = time
            self.ticked = True

    def _do_reset(self):
        self.ticked = False


class MultiScalarPort(TimeSeriesPort):
    """
    Like scalar port but accepts multiple values at the same time
    """
    def __init__(self, unit, name, history=0):
        TimeSeriesPort.__init__(self, unit, name, history=history)
        self._image_time = None
        self._image = None

    def value(self):
        if self._image is not None:
            return self._image

        raise ValueError('No tick in port %s', self._name)

    def _copy_value(self):
        copied = self._image[:]
        return copied

    def empty(self):
        return self._image is None

    def _do_tick(self, time, value):
        if value is not None:
            if time == self._image_time:
                if self.ticked:
                    # multiple ticks in the same time instance
                    self._image.append(value)
                else:
                    raise ValueError('Same time value after the tick was consumed for port %s', self._name)
            else:
                # new time instance
                self._image = [value]

            self._image_time = time
            self.ticked = True

    def _do_reset(self):
        self.ticked = False


class DictPort(TimeSeriesPort):
    def __init__(self, unit, name, history=0):
        TimeSeriesPort.__init__(self, unit, name, history=history)
        self._image = {}
        self.active = {}
        self.added = {}

    def value(self):
        if len(self._image) == 0:
            raise ValueError('No tick in port %s', self._name)

        return self._image

    def _copy_value(self):
        return self._image.copy()

    def empty(self):
        return len(self._image) == 0

    def _do_tick(self, time, value):
        for k, v in value.items():
            if v is not None:
                if k not in self._image:
                    self.added[k] = v
                else:
                    self.active[k] = v
                self._image[k] = v
                self.ticked = True

    def _do_reset(self):
        self.ticked = False
        self.active.clear()
        self.added.clear()


class OutputPort(object):
    """
    An output port corresponds to a single output of a TimeSeriesUnit
    where any consuming input ports attach themselves as listeners.

    Multiple values can be ticked into an OutputPort in a single
    cycle, the consuming input port is responsible for handling
    (or rejecting) them
    """

    def __init__(self, owner):
        self._owner = owner
        self._consumers = set()  # ports that consume us
        self._consumer_units = set()  # set of units owning those ports
        self._tick = []

    def add_consumer(self, consumer):
        """
        Adds an observer to the node that will be notified
        when data arrives
        """
        self._consumers.add(consumer)

    def remove_consumer(self, consumer):
        self._consumers.remove(consumer)

    def tick(self, value):
        self._tick.append(value)

    def notify_consumers(self, timestamp):
        for value in self._tick:
            for consumer in self._consumers:
                consumer.tick(timestamp, value)
        self._tick = []

    def owner(self):
        return self._owner


def create_input_port(name, ts_type, owner, default=None, history=0):
    """
    Factory method for input ports
    """
    if default is not None and ts_type is not TS_SCALAR:
        raise ValueError('Defaults only supported for TS_SCALAR ports')

    if ts_type == TS_SCALAR:
        return ScalarPort(owner, name, default, history=history)
    if ts_type == TS_MULTI_SCALAR:
        return MultiScalarPort(owner, name, history=history)
    elif ts_type == TS_DICT:
        return DictPort(owner, name, history=history)
    else:
        raise KeyError('Unknown timeseries type %s for input %s' % (ts_type, name))


class TimeSeriesUnit(object):
    """
    Base class for time series units
    """

    def __init__(self, function, ts_def_spec, decorated=False, name=None):
        self._function = function  # the action function
        self._engine = None
        self._wired = False
        self._ts_def_input_spec = ts_def_spec.get(TS_INPUT, {})
        self._ts_def_output_names = ts_def_spec.get(TS_OUTPUT, set())

        self._arg_cache = []  # the cache of time series ports or values for position arguments
        self._kwarg_cache = {}  # the cache of time series ports or values for keyword arguments
        self.rank = 0
        self._input_ports = {}  # the union of arg_cache and kwarg_cache for looking ports up by name
        self._init_alarms = {}
        self._output_ports = {}
        self._dependencies = set()

        self._decorated = decorated
        self._varargs = None  # create ports dynamically based on the input timeseries
        self._name = name or self.__class__.__name__  # For debugging

        # static checks for the shape of the action function and the ts_def_spec
        if self._function is not None:
            (args, varargs, kwargs, defaults) = inspect.getargspec(self._function)
            if args[0] == 'self':
                args = args[1:]
            if len(args) == 0 or args[0] != 'timestamp':
                raise ValueError('the first argument of a timeseries function must be '
                                 'a positional argument called timestamp')
            args = args[1:]
            if kwargs is not None:
                raise ValueError('timeseries functions that use kwargs are unsupported')

            if varargs is not None:
                if len(args) > 0:
                    raise ValueError('timeseries function that use varargs cannot have any other arguments')
                else:
                    self._varargs = varargs

            defaults = {} if defaults is None else defaults
            # the last n args that have defaults are the keyword args
            self._kwargs = dict(zip(args[-len(defaults):], defaults)) if len(defaults) > 0 else {}
            # the remaining are the positional args
            self._args = args[0:len(args) - len(defaults)] if len(args) >= len(defaults) else []
        else:
            # empty provider
            self._args = []
            self._kwargs = {}

    def __call__(self, *arg_values, **kwarg_values):
        """
        Calling the TimeSeriesUnit has the effect of wiring it in to
        the current circuit by subscribing it's inputs to consume ticks
        from the provided time series. If the argument has a default value
        and no explicit value has been provided, if the argument is defined as
        a time series input we will attempt to create a non ticking time series port
        otherwise the value will be passed in as is
        """
        if self._wired:
            raise ValueError('Unit has already been wired')
        self._wired = True

        # the number of positional timeseries function arguments must match the number
        # of arguments passed in unless we are using varargs
        if self._varargs is not None:
            self._args = [self._varargs] * len(arg_values)

        if len(arg_values) != len(self._args):
            raise ValueError('all positional arguments must be passed a value')

        for arg, arg_value in zip(self._args, arg_values):
            arg_spec = self._ts_def_input_spec[arg]
            if isinstance(arg_spec, str):
                arg_type = arg_spec
                history = 0
            else:
                if len(arg_spec) == 2:
                    # add historical access
                    arg_type = arg_spec[0]
                    history = arg_spec[1]
                elif len(arg_spec) == 3:
                    # schedule initial alarm
                    arg_type = arg_spec[0]
                    self._init_alarms[arg] = arg_spec[1:3]
                    history = 0
                else:
                    raise ValueError('could not understand input port specification %s' % repr(arg_spec))

            if isinstance(arg_value, OutputPort):
                # wire in the output port of another unit
                self._dependencies.add(arg_value.owner())

                input_port = create_input_port(arg, arg_type, self, history=history)
                arg_value.add_consumer(input_port)
                self._arg_cache.append(input_port)
                self._input_ports[arg] = input_port
            else:
                # TODO support wiring in a constant value
                raise ValueError('Unsupported non output port argument for input %s' % arg)

        for kwarg, kwarg_default in self._kwargs.items():
            kwarg_value = kwarg_values.get(kwarg, kwarg_default)

            kwarg_spec = self._ts_def_input_spec[kwarg]
            if isinstance(kwarg_spec, str):
                kwarg_type = kwarg_spec
                history = 0
            else:
                if len(kwarg_spec) == 2:
                    # add historical access
                    kwarg_type = kwarg_spec[0]
                    history = kwarg_spec[1]
                elif len(kwarg_spec) == 3:
                    kwarg_type = kwarg_spec[0]
                    self._init_alarms[kwarg] = kwarg_spec[1:3]
                    history = 0
                else:
                    raise ValueError('could not understand input port specification %s' % repr(kwarg_spec))

            if isinstance(kwarg_value, OutputPort):
                self._dependencies.add(kwarg_value.owner())

                input_port = create_input_port(kwarg, kwarg_type, self, history=history)
                kwarg_value.add_consumer(input_port)
                self._kwarg_cache[kwarg] = input_port
                self._input_ports[kwarg] = input_port
            else:
                if kwarg in self._ts_def_input_spec:
                    input_port = create_input_port(kwarg, kwarg_type,
                                                   self, default=kwarg_value)
                    self._kwarg_cache[kwarg] = input_port
                    self._input_ports[kwarg] = input_port
                else:
                    raise ValueError('Keyword argument %s is not in the spec' % kwarg)

        self._output_ports = {}

        if len(self._ts_def_output_names) == 0:
            # by default create a single output port called _output
            # and return the port directly to be used for wiring
            output_port = OutputPort(self)
            self._output_ports['_output'] = output_port
            return output_port
        else:
            # if the ports have been named explictly, return
            # a dict of port name to port
            for output_name in self._ts_def_output_names:
                self._output_ports[output_name] = OutputPort(self)
            return self._output_ports

    def activate(self):
        """
        Call the actual time series consumers after any inputs
        have ticked in for this time instance
        """
        timestamp = self._engine.now()

        if self._decorated:
            self._function(self, timestamp, *self._arg_cache, **self._kwarg_cache)
        else:
            self._function(timestamp, *self._arg_cache, **self._kwarg_cache)

        # notify consumers
        for output_port in self._output_ports.values():
            output_port.notify_consumers(timestamp)

        # reset the ports
        for input_ts in self._arg_cache:
            input_ts.reset()

        for input_ts in self._kwarg_cache.values():
            input_ts.reset()

    def alarm(self, alarm_time, input_port, value=ALARM_TICK):
        if isinstance(input_port, str):
            input_port = self._input_ports[input_port]
        self._engine.alarm(alarm_time, input_port, alarm_value=value)

    def async_alarm(self, input_port, value=ALARM_TICK):
        if isinstance(input_port, str):
            input_port = self._input_ports[input_port]
        self._engine.async_alarm(input_port, alarm_value=value)

    def start(self, engine, start_time, end_time):
        """
        Called before an engine run to give each unit a chance to start
        by loading any data, setting the first alarms etc
        """
        self._engine = engine
        for input_name, alarm_info in self._init_alarms.items():
            engine.alarm(alarm_info[0], self._input_ports[input_name], alarm_info[1])
        self._do_start(start_time, end_time)

    def _do_start(self, start_time, end_time):
        """
        Override in custom TimeSeriesUnit implementations to perform start up logic
        """
        pass

    def stop(self, stop_time):
        """
        Called after an engine finishes the run.
        """
        self._do_stop(stop_time)

    def _do_stop(self, stop_time):
        """
        Override in custom TimeSeriesUnit implementations to perform shutdown logic
        """
        pass

    def dependencies(self):
        return self._dependencies

    def schedule(self):
        self._engine.event(self)


class TimeSeriesRecorder(TimeSeriesUnit):
    """
    Receives the output from a timeseries unit and returns it
    as a pandas timeseries
    """
    def __init__(self, input_name='tick', input_type=TS_MULTI_SCALAR, debug_name='', copy=True, record_only_last=False):
        """

        :param input_name:
        :param input_type:
        :param debug_name: for debug purpose only
        :param copy: deepcopy each passed object
        :param record_only_last: record only the last element of the timeserie
        """

        spec = {TS_INPUT: {input_name: input_type}}
        TimeSeriesUnit.__init__(self, self.action, spec)
        # init at runtime when we know the input type
        self._timestamps = None
        self._values = None
        self.debug_name = debug_name
        self.copy = copy
        self.record_only_last = record_only_last

    def action(self, timestamp, tick):
        values = tick.value()

        if self._timestamps is None:
            if isinstance(values[0], dict):
                # if the first tick to arrive is a dict we will guess
                # that we need to record separate timeseries
                self._timestamps = defaultdict(list)
                self._values = defaultdict(list)
            else:
                self._timestamps = []
                self._values = []

        if tick.ticked:
            for value in values:
                if isinstance(self._timestamps, dict):
                    # if the first value we saw was a dict we are going
                    # to assume they all are
                    for key, val in value.items():
                        if self.copy:
                            store = copy.deepcopy(val)
                        else:
                            store = val
                        if self.record_only_last and len(self._values[key]) > 0:
                                self._values[key][0] = store
                                self._timestamps[key][0] = timestamp
                        else:
                            self._values[key].append(val)
                            self._timestamps[key].append(timestamp)
                else:
                    if self.copy:
                        store = copy.deepcopy(value)
                    else:
                        store = value
                    if self.record_only_last and len(self._values) > 0:
                        self._timestamps[0] = timestamp
                        self._values[0] = store
                    else:
                        self._timestamps.append(timestamp)
                        self._values.append(store)

    def ts(self):
        if isinstance(self._timestamps, dict):
            result = {}
            for key in self._timestamps.keys():
                result[key] = Series(self._values[key], self._timestamps[key])
            return result

        return Series(self._values, self._timestamps)


def tsdef(spec):
    """
    Decorator factory function for timeseries consumers/producer
    """

    def _time_series_consumer(function):
        unit = TimeSeriesUnit(function, spec, decorated=True)
        return unit

    return _time_series_consumer


def _discover_and_rank_units(result, walked, unit):
    """
    Recursively walk dependency graph backwards to get all units the
    unit depends on, ranking them in the process
    """
    walked.add(unit)
    deps = unit.dependencies()
    if len(deps) > 0:
        for dep in deps:
            result.add(dep)
            if dep not in walked:
                _discover_and_rank_units(result, walked, dep)

        unit.rank = max([dep.rank for dep in deps]) + 1
    else:
        unit.rank = 0


def tsrun(circuit, engine, start_time, end_time=None, record_outputs=False, recorders_options=None):
    """
    Wire the circuit, attach the engine and run it, returning
    the top level output timeseries
    """
    # get the top level outputs
    outputs = circuit()

    if isinstance(outputs, OutputPort):
        outputs = {'_output': outputs}

    # optionally attach recorders to the outputs so we can extract timeseries of their values
    if record_outputs:
        recorders, top_level_outputs = _attach_recorders(outputs, recorders_options=recorders_options)
    else:
        top_level_outputs = outputs

    units = set()
    for top_level_output in top_level_outputs.values():
        result = set()
        top_level_unit = top_level_output.owner()
        _discover_and_rank_units(result, set(), top_level_unit)
        units.update(result)
        units.add(top_level_unit)

    # start them
    for unit in units:
        unit.start(engine, start_time, end_time)

    # go
    try:
        engine.start(start_time, end_time)
    finally:
        for unit in units:
            unit.stop(engine.now())

    # collect and return the outputs
    if record_outputs:
        return {name: recorder.ts() for (name, recorder) in recorders.items()}
    else:
        return None


def _attach_recorders(outputs, recorders_options=None):
    """
    Attach recorder units to the top level outputs of the circuit
    and return the recorders and their outputs.

    :param recorders_options: Override behaviour of recorders
                              {'name': (copy, record_only_last)} default is copy=True, record_only_last=False
    """

    if recorders_options is None:
        recorders_options = dict()

    recorders = {}
    recorder_outputs = {}

    for name, output in outputs.items():
        copy, record_only_last = recorders_options[name] if name in recorders_options else (True, False)
        recorder = TimeSeriesRecorder(debug_name=name, copy=copy, record_only_last=record_only_last)
        recorders[name] = recorder
        recorder_outputs[name] = recorder(output)

    return recorders, recorder_outputs
