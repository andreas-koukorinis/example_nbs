import json
import logging
from collections import defaultdict
from datetime import timedelta
from logging import getLogger

from cassandra import ConsistencyLevel
from cassandra.query import BatchStatement
import pandas as pd

from sgmtradingcore.core.definitions import (
    TimeSeriesUnit,
    TS_INPUT,
    TS_SCALAR,
    TS_MULTI_SCALAR, TS_DICT, TS_OUTPUT)

import itertools

from sgmtradingcore.core.ts_log import get_logger


class TimeSeriesProvider(TimeSeriesUnit):
    """
    Receives a pandas timeseries and ticks it as its output.
    """
    def __init__(self, timeseries):
        self._ts = timeseries
        self._index = 0
        spec = {TS_INPUT: {'tick': TS_SCALAR}}
        TimeSeriesUnit.__init__(self, self.action, spec)

    def _do_start(self, start_time, end_time):
        self._ts = self._ts.truncate(before=start_time, copy=False)
        if len(self._ts) == 0:
            return
        ts_start = self._ts.index[0].to_datetime()
        self._engine.alarm(ts_start, self._input_ports['tick'])

    def action(self, timestamp, tick=None):
        if tick.ticked:
            values = self._ts.ix[timestamp]
            if not isinstance(values, pd.Series):
                len_values = 1
                new_values = [values]
            else:
                len_values = len(values)
                new_values = values.tolist()

            if self._index + len_values - 1 < len(self._ts) - 1:
                self._index += len_values
                ts_next = self._ts.index[self._index].to_datetime()
                tick.alarm(ts_next, 1)

            for value in new_values:
                self._output_ports['_output'].tick(value)


def timeseries_provider(timeseries):
    provider = TimeSeriesProvider(timeseries)
    return provider()


class DictTimeseriesProvider(TimeSeriesProvider):
    """
    Receives a list of dicts and extracts the time
    using a key function to tick out as a timeseries
    """

    def __init__(self, dicts, key_func):
        dicts = sorted(dicts, key=key_func)
        timestamps = [key_func(the_dict) for the_dict in dicts]
        ts = pd.Series(dicts, index=timestamps)
        TimeSeriesProvider.__init__(self, ts)


def dict_timeseries_provider(dicts, key_func):
    provider = DictTimeseriesProvider(dicts, key_func)
    return provider()


class TimeSeriesLogger(TimeSeriesUnit):
    """
    Logs input as it ticks
    """
    def __init__(self, prefix, ts_type=TS_SCALAR):
        self._logger = getLogger(prefix)
        spec = {TS_INPUT: {'input': TS_SCALAR}}
        TimeSeriesUnit.__init__(self, self.action, spec)

    def action(self, timestamp, input):
        if input.ticked:
            self._logger.info(input.value())


def timeseries_logger(input, prefix, ts_type=TS_SCALAR):
    logger = TimeSeriesLogger(prefix, ts_type)
    return logger(input)


class DirectPush(TimeSeriesUnit):
    """
    Receives a tick and pushes it directly to its consumers. This unit does some unsafe
    operations and should not be used outside of internal circuitry.
    """

    def __init__(self, name=None):
        spec = {
            TS_INPUT: {'value': TS_MULTI_SCALAR},
        }
        TimeSeriesUnit.__init__(self, self.action, spec, name=name)
        self._last_action_timestamp = None
        self._delay_ms = 1

    def action(self, timestamp, value=None):
        if value.ticked:
            for item in value.value():
                self._output_ports['_output'].tick(item)
            self._last_action_timestamp = timestamp

    def push_value(self, value):
        ts = self._engine.now()
        # Avoid pushing in the same timestamp if action() has been already called.
        if ts != self._last_action_timestamp:
            self._input_ports['value'].tick(ts, value)
        else:
            # Push it later to avoid ticking twice the same port in the same timestamp
            return self.alarm(
                timedelta(milliseconds=self._delay_ms),
                self._input_ports['value'], value)


class DelayedTickPush(TimeSeriesUnit):
    """
    Receives a tick and pushes it back out after a given delay. Useful for breaking loops
    and the chicken & egg problem that comes with them
    """
    def __init__(self, delay=1, name=None):
        spec = {
            TS_INPUT: {'delayed': TS_MULTI_SCALAR},
        }
        TimeSeriesUnit.__init__(self, self.action, spec, name=name)
        self._delay = delay

    def action(self, timestamp, delayed=None):
        if delayed.ticked:
            for item in delayed.value():
                self._output_ports['_output'].tick(item)

    def push_value(self, value, delay_ms=None):
        if delay_ms is None:
            delay_ms = self._delay

        return self.alarm(
            timedelta(milliseconds=delay_ms),
            self._input_ports['delayed'], value)


class DelayedTickPusher(TimeSeriesUnit):
    """
    Receives an input timeseries and pushes all ticks to the provided push port
    """
    def __init__(self, push_port, name=None):
        spec = {
            TS_INPUT: {'input_': TS_MULTI_SCALAR},
        }
        TimeSeriesUnit.__init__(self, self.action, spec, name=name)
        self._push_port = push_port

    def action(self, timestamp, input_):
        if input_.ticked:
            for item in input_.value():
                self._push_port.push_value(item)


class EmptyProvider(TimeSeriesUnit):
    def __init__(self):
        spec = {
            TS_INPUT: {'input_': TS_MULTI_SCALAR},
        }
        TimeSeriesUnit.__init__(self, self.action, spec)

    def action(self, timestamp, *input_):
        pass

    def subscribe(self, *args, **kwargs):
        pass

    def unsubscribe(self, *args, **kwargs):
        pass


def empty_provider():
    provider = EmptyProvider()
    return provider()


class DictMultiOutputDemultiplexer(TimeSeriesUnit):
    """
    Demultiplex an incoming TS_DICT into multiple outputs based on a key.
    The default key is simply the dict key but an arbitrary function can be passed in
    to produce the key.

    For strategies that run multiple instances of themselves, such as market making
    which is instantiated once per orderbook/sticker, this can be used to take in all
    market data updates on a single port and fan them out to multiple ports.
    """

    def __init__(self, keys, key_func=lambda k, v: k, strict_keys=True):
        outputs = {key: TS_SCALAR for key in keys}
        spec = {
            TS_INPUT: {'input_': TS_DICT},
            TS_OUTPUT: outputs
        }
        TimeSeriesUnit.__init__(self, self.action, spec)
        self._key_func = key_func
        self._strict_keys = strict_keys
        self._logger = get_logger(__name__)

    def action(self, timestamp, input_):
        for key, value in itertools.chain(input_.added.items(), input_.active.items()):
            output_key = self._key_func(key, value)
            if output_key not in self._output_ports and not self._strict_keys:
                self._logger.warn(timestamp, 'Ignoring update %s for unknown key %s' % (value, output_key))
                continue
            self._output_ports[output_key].tick(value)


class MultiScalarMultiOutputDemultiplexer(TimeSeriesUnit):
    """
    Similar to above for TS_MULTI_SCALAR inputs
    """

    def __init__(self, keys, key_func=lambda v: v, strict_keys=True):
        outputs = {key: TS_SCALAR for key in keys}
        spec = {
            TS_INPUT: {'input_': TS_MULTI_SCALAR},
            TS_OUTPUT: outputs
        }
        TimeSeriesUnit.__init__(self, self.action, spec)
        self._key_func = key_func
        self._strict_keys = strict_keys
        self._logger = get_logger(__name__)

    def action(self, timestamp, input_):
        for value in input_.value():
            output_key = self._key_func(value)
            if output_key not in self._output_ports and not self._strict_keys:
                self._logger.warn(timestamp, 'Ignoring update %s for unknown key %s' % (value, output_key))
                continue
            self._output_ports[output_key].tick(value)


class MultiScalarMultiInputCombiner(TimeSeriesUnit):
    """
    Combines multiple TS_MULTI_SCALAR timeseries into a single one
    """

    def __init__(self):
        spec = {
            TS_INPUT: {'input_': TS_MULTI_SCALAR},
        }
        TimeSeriesUnit.__init__(self, self.action, spec)

    def action(self, timestamp, *input_):
        for input_ts in input_:
            if input_ts.ticked:
                for input_ts_val in input_ts.value():
                    self._output_ports['_output'].tick(input_ts_val)


_INSERT_QUERY = """
INSERT INTO analysisdata
(analysis_key, time, tick)
VALUES
(?, ?, ?)
USING TTL 2592000
"""


class TimeSeriesCassandraPublisher(TimeSeriesUnit):
    """
    Publishes every update of a timeseries to Cassandra
    """

    def __init__(self, trading_user, strategy_id, strategy_run_id, ts_name, cassandra,
                 jsonifier=lambda x: json.dumps(x), name=None):

        spec = {
            TS_INPUT: {'input_ts': TS_MULTI_SCALAR},
        }
        TimeSeriesUnit.__init__(self, self.action, spec)
        self._trading_user = trading_user
        self._strategy_id = strategy_id
        self._strategy_run_id = strategy_run_id
        self._ts_name = ts_name
        self._cassandra = cassandra
        self._cassandra.connect(keyspace='strategydata')
        self._insert_query = cassandra.session.prepare(_INSERT_QUERY)

        # for debugging
        self._name = name

        self._jsonifier = jsonifier

    def _log_cassandra_error(self, exc):
        logging.error('Cassandra operation failed: %s', exc)

    def action(self, timestamp, input_ts):
        if input_ts.ticked:

            key = '%s.%s.%s.%s' % (
                self._trading_user, self._strategy_id, self._strategy_run_id, self._ts_name)

            for input_ts_val in input_ts.value():
                if isinstance(input_ts_val, dict):
                    for name, val in input_ts_val.items():
                        json_out = self._jsonifier(val)
                        params = ('%s.%s' % (key, name), timestamp, json_out)
                        future = self._cassandra.session.execute_async(self._insert_query, params)
                        future.add_errback(self._log_cassandra_error)
                else:
                    json_out = self._jsonifier(input_ts_val)
                    params = (key, timestamp, json_out)
                    future = self._cassandra.session.execute_async(self._insert_query, params)
                    future.add_errback(self._log_cassandra_error)


class AnalysisTimeSeriesCassandraPublisher(TimeSeriesCassandraPublisher):
    """
    This class will always store results as a list
    and will group ticks with the same key / timestamp rather than only accepting the final one received
    """
    def action(self, timestamp, input_ts):

        if input_ts.ticked:

            items_to_insert = defaultdict(lambda: [])

            key = '%s.%s.%s.%s' % (
                self._trading_user, self._strategy_id, self._strategy_run_id, self._ts_name)

            for input_ts_val in input_ts.value():
                if isinstance(input_ts_val, dict):
                    for name, val in input_ts_val.items():
                        cass_key = '%s.%s' % (key, name)
                        items_to_insert[(cass_key, timestamp)].append(val)
                else:
                    items_to_insert[(key, timestamp)].append(input_ts_val)

            for (cass_key, timestamp), value in items_to_insert.iteritems():
                params = (cass_key, timestamp, self._jsonifier(value))
                future = self._cassandra.session.execute_async(self._insert_query, params)
                future.add_errback(self._log_cassandra_error)
