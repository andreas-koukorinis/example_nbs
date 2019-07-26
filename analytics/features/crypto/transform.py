import datetime
from collections import OrderedDict

import pandas as pd

from sgmtradingcore.analytics.features.feature import InfiniteTimeSeriesFeature
from sgmtradingcore.analytics.features.request import FeatureRequest


class TransformationFeature(InfiniteTimeSeriesFeature):
    """
    A general purpose base class models allows a feature to accept input from data
    (LOB or trades) or another feature, perform transformation on these data and stores.
    In other ways, it allows for calculation of features both on raw data as well as
    chaining on other features.

    Attributes:
    -----------------------------
    _columns (list of strings): The columns used to extract data from the dataframes returned
                       from arctic / other features.
    _input (FeatureRequest): Controls the source of data input. Possible sources are arctic,
                            raw data, other features.
    """
    REQUIRED_PARAMETERS = {'input', 'columns'}

    def __init__(self, runner, params, **kwargs):
        """This feature takes in /one/ feature and applies some fixed period transformation to it"""

        super(TransformationFeature, self).__init__(runner, params, **kwargs)
        self._input = params['input']
        self._columns = params['columns']

    @classmethod
    def _check_parameters(cls, params):
        if not isinstance(params['input'], FeatureRequest):
            raise TypeError('Input for {} should be a FeatureRequest'.format(cls.__name__))

        if not isinstance(params['columns'], list):
            raise ValueError('Columns input for {} should be a list'.format(cls.__name__))

    def _input_feature(self, start, end):
        feature_request = self._input

        ticker_feature = self._runner.get_merged_dataframes([feature_request], start_dt=start, end_dt=end)
        if ticker_feature.empty:
            return ticker_feature
        else:
            return ticker_feature[self._columns]

    def _compute(self, start_dt, end_dt):
        raise NotImplementedError

    @property
    def columns(self):
        return self._columns


class AggregationFeature(TransformationFeature):
    REQUIRED_PARAMETERS = {'clock'}

    def __init__(self, runner, params, storage=None, storage_period=None, **kwargs):
        """
        A general purpose base class that takes
        1) a clock (i.e. a division of time into non-overlapping intervals)
        2) An input feature (e.g LOB or trades)
        And returns some sort of aggregation over each clock period.

        :param runner:
        :param params:
        :param storage:
        :param storage_period:
        :param kwargs:
        """

        self._storage_period = storage_period or self.recommended_storage_period(runner, params)

        super(AggregationFeature, self).__init__(runner, params, storage=storage, storage_period=storage_period,
                                                 **kwargs)

    @classmethod
    def recommended_storage_period(cls, runner, params):
        clock_params = params['clock'].feature_params(runner)
        clock_class = params['clock'].get_feature_class(runner)
        return clock_class.recommended_storage_period(clock_params)

    @classmethod
    def _check_parameters(cls, params):
        if not isinstance(params['clock'], FeatureRequest):
            raise TypeError('Clock for {} should be a FeatureRequest'.format(cls.__name__))

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'clock': FeatureRequest('FixedPeriodClock', {'frequency': '1H'})}

    def _clock_feature(self, start, end):
        feature_request = self._params['clock']
        feature = self._runner.get_merged_dataframes([feature_request], start_dt=start, end_dt=end)
        if 'n_ticked' not in feature.columns or 'previous' not in feature.columns:
            raise ValueError('Clock should have n_ticked and previous columns')
        return feature[['n_ticked', 'previous']]

    def _compute_ticks(self, groups):
        # Use _input_feature
        # Start of tick is tick.previous,
        # end of tick is tick.Index
        raise NotImplementedError

    def _compute(self, start_dt, end_dt):

        clock = self._clock_feature(start_dt, end_dt)
        if clock.empty:
            return self.get_empty_feature()

        input = self._input_feature(clock.iloc[0]['previous'], clock.iloc[-1].name)
        clock['tick'] = clock.index.to_series(keep_tz=True)

        if len(input) == 0:
            return self.get_empty_feature()
        merged_data = pd.merge_asof(input, clock.set_index('previous', drop=False), left_index=True, right_index=True)

        return self._compute_ticks(merged_data.groupby('tick'))


class RecursiveAggregationFeature(AggregationFeature):
    """
    This type of feature aggregates an input feature over clock intervals, but each tick is a function of the last tick
    """
    REQUIRED_PARAMETERS = {'start'}
    RECURSIVE = True

    @classmethod
    def _check_parameters(cls, params):
        if not isinstance(params['start'], datetime.datetime):
            raise ValueError('Start should be a datetime')

    def _compute_recursive_tick(self, clock_tick, previous_tick, data):
        # Use _input_feature
        # Start of interval is tick.previous
        # end of interval is tick.Index
        raise NotImplementedError

    def _compute(self, start_dt, end_dt):

        if end_dt < self._params['start']:
            msg = 'Can''t calculate this range as {} is before the clock start {}'
            raise ValueError(msg.format(end_dt, self._params['start']))

        previous_tick = self._storage.load_previous_tick(start_dt)

        if previous_tick is None:
            previous_tick_end = self._params['start']
        else:
            previous_tick_end = previous_tick.name + datetime.timedelta(microseconds=1)

        # Calculate ticks from the previous ticks until the interval that we are currently interested in
        request = [FeatureRequest.from_feature(self)]
        if previous_tick_end != start_dt:
            self._runner.compute_dataframes(request, previous_tick_end, start_dt)
        new_previous_tick = self._storage.load_previous_tick(start_dt)
        if new_previous_tick is None or new_previous_tick.empty:
            new_previous_tick = previous_tick
        else:
            new_previous_tick = previous_tick

        # Should be up-to-date now, so get the updated latest tick

        # If there is still no previous tick then we have to start from the beginning
        if new_previous_tick is None:
            start = self._params['start']
        elif new_previous_tick.empty:
            raise ValueError('This should not be possible')
        else:
            start = new_previous_tick.name + datetime.timedelta(microseconds=1)

        clock = self._clock_feature(start, end_dt)
        if len(clock) == 0:
            # There are no ticks in this range, so nothing to do
            return None

        input = self._input_feature(clock.iloc[0]['previous'], clock.iloc[-1].name)
        clock['tick'] = clock.index.to_series(keep_tz=True)

        merged_data = pd.merge_asof(input, clock.set_index('previous', drop=False),
                                    left_index=True, right_index=True)

        ticks = []
        for tick, df in merged_data.groupby('tick'):
            clock_tick = clock.loc[tick]
            new_previous_tick = pd.Series(self._compute_recursive_tick(clock_tick, new_previous_tick, df),
                                          name=tick)
            ticks.append(new_previous_tick)

        return pd.DataFrame([t for t in ticks if end_dt > t.name >= start_dt])
