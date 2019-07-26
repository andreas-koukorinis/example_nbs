import math
import datetime as dt
import pandas as pd
import pytz
from pandas.tseries.frequencies import to_offset

from sgmtradingcore.analytics.features.crypto.market_data import CryptoDataSource
from sgmtradingcore.analytics.features.feature import InfiniteTimeSeriesFeature
from sgmtradingcore.analytics.features.request import FeatureRequest
from sgmtradingcore.analytics.features.storage import EmptyFeatureStorage, MongoInfiniteTimeseriesStorage
from stratagemdataprocessing.crypto.enums import get_first_valid_datetime
from stratagemdataprocessing.crypto.enums import CryptoCrossSide


class ClockFeature(InfiniteTimeSeriesFeature):
    REQUIRED_PARAMETERS = {'start'}

    CLOCK_TICKER_NAME = 'clock'

    def __init__(self, runner, params, storage=None, storage_period=None, **kwargs):
        """
        A clock feature should return a dataframe with a timestamp index
        the values of which indicate when the clock has ticked.

        Each clock needs to implement _compute_recursion which takes the previous clock tick and returns all ticks
        needed for the range requested

        Implementation Notes:
        The index of each bar is the /end/ of the tick (for storage reasons)

        i.e. if we request a clock from s -> e we need to know the tick that started before s and ended after s
        so that we can calculate subsequent ticks.

        each tick should contain all the information needed to calculate subsequent ticks only given the market data
        (or whatever it requires) for the interval that we are interested in.

        Clocks should also contain:
        n_ticked: which is >1 if the clock ticks more than once at a given time
        previous: the time of the previous tick

        Ticker is included in the params rather than the call to _compute as we need
        to be careful with start times and adding new data etc.
        Also, this will be used as an index for other features.

        :param runner:
        :param params:
        :param storage:
        :param storage_period:
        :param kwargs:
        """

        super(ClockFeature, self).__init__(runner,
                                           params,
                                           storage=storage,
                                           storage_period=storage_period,
                                           **kwargs)

    @classmethod
    def _default_parameters(cls, params, runner):
        return {}

    @classmethod
    def get_empty_feature(cls):
        return pd.DataFrame([], columns=['previous', 'n_ticked'])

    def _compute(self, start_dt, end_dt):
        return super(ClockFeature, self)._compute(start_dt, end_dt)

    def get_df(self, start_dt, end_dt, repopulate=False):
        return super(ClockFeature, self).get_df(start_dt, end_dt, repopulate=repopulate)

    @classmethod
    def recommended_storage_period(cls, params):
        """
        Given params for the clock, this suggests what storage period should be used for any feature based on this clock

        :param params:
        :return:
        """
        raise NotImplementedError('Implement in subclass')

    def _check_computation(self, df, start, end):

        df = super(ClockFeature, self)._check_computation(df, start, end)

        if 'n_ticked' not in df:
            raise ValueError('n_ticked should be in clock columns')

        if 'previous' not in df:
            raise ValueError('previous should be in clock columns')

        return df


class FixedPeriodClock(ClockFeature):
    REQUIRED_PARAMETERS = {'frequency', 'offset'}

    def __init__(self, runner, params):
        """
        This Clock returns timestamps with fixed intervals from a start datetime.
        :param runner:
        :param params:
        """
        storage = EmptyFeatureStorage(self)

        super(FixedPeriodClock, self).__init__(runner, params, storage, None)

        if params['offset'] > self.frequency:
            raise ValueError(
                'Offset {} should be below the frequency {} for {}'.format(params['offset'], params['frequency'],
                                                                           self.__class__.__name__))

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'start': dt.datetime(1970, 1, 1, tzinfo=pytz.UTC),
                'frequency': '1H',
                'offset': dt.timedelta(0)}

    @classmethod
    def _check_parameters(cls, params):
        if params['frequency'][-1] not in ['D', 'H', 'T', 'S']:
            raise ValueError("Mis specified timeseries offset")

    @staticmethod
    def str_to_timedelta(f):
        return pd.Timedelta(to_offset(f)).to_pytimedelta()

    @property
    def frequency(self):
        return self.str_to_timedelta(self._params['frequency'])

    def _compute(self, start_dt, end_dt):
        f = self.frequency

        interval = (start_dt - self._params['start']).total_seconds() / f.total_seconds()
        rounded_start = self._params['start'] + (int(math.floor(interval)) - 1) * f + self._params['offset']
        t = rounded_start

        times = [t]
        while t < end_dt:
            t += f
            times.append(t)

        timestamps = pd.DataFrame({'timestamp': times[1:], 'previous': times[:-1]}).set_index('timestamp')
        timestamps['n_ticked'] = 1

        return timestamps

    @classmethod
    def recommended_storage_period(cls, params):

        valid_storage_periods = [dt.timedelta(weeks=52),
                                 dt.timedelta(weeks=4),
                                 dt.timedelta(weeks=1),
                                 dt.timedelta(days=7),
                                 dt.timedelta(days=3),
                                 dt.timedelta(days=1),
                                 dt.timedelta(hours=6)]

        for i, p in reversed(list(enumerate(valid_storage_periods))):
            if p > cls.str_to_timedelta(params['frequency']):
                # Choose the frequency two bigger than the frequency
                return valid_storage_periods[max(i - 1, 0)]

        raise ValueError('Invalid frequency for storage')


class RecursiveClockFeature(ClockFeature):
    RECURSIVE = True

    """
    This type of clock requires the result of the previous tick in order to calculate the next.
    All information required for the next tick should be stored in the result of the previous.
    """

    @classmethod
    def recommended_storage_period(cls, params):
        raise NotImplementedError

    @classmethod
    def _check_parameters(cls, params):
        if not isinstance(params['start'], dt.datetime):
            raise ValueError('{} start should be a datetime'.format(cls.__name__))
        if params['start'].tzinfo is None:
            raise ValueError('{} start should have a timezone'.format(cls.__name__))

    def _compute_recursion(self, previous_tick, start_dt, end_dt):
        raise NotImplementedError('Implement in subclass')

    def _compute(self, start_dt, end_dt):
        previous_tick = self._storage.load_previous_tick(start_dt)

        if previous_tick is None:
            previous_tick_end = self._params['start']
        else:
            previous_tick_end = previous_tick.name

        # Calculate ticks from the previous ticks until the interval that we are currently interested in
        request = [FeatureRequest.from_feature(self)]
        if previous_tick_end != start_dt:
            self._runner.compute_dataframes(request, previous_tick_end, start_dt)
        new_previous_tick = self._storage.load_previous_tick(start_dt)

        if new_previous_tick is None or new_previous_tick.empty:
            new_previous_tick = previous_tick

        # Should be up-to-date now, so get the updated latest tick
        # If there is still no previous tick then we have to start from the beginning
        if new_previous_tick is None:
            start = self._params['start']
        elif new_previous_tick.empty:
            raise ValueError('This should not be possible')
        else:
            start = new_previous_tick.name + dt.timedelta(microseconds=1)

        return self._compute_recursion(previous_tick, start, end_dt)


class FixedVolumeClock(RecursiveClockFeature):
    REQUIRED_PARAMETERS = {'volume', 'units', 'ticker', 'source', 'offset'}

    def __init__(self, runner, params, **kwargs):
        """
        Ticks the clock every time `volume` of `units` is matched in the ticker.
        :param runner:
        :param params:
        start: the first datetime to start from
        volume: the amount in each tick
        units: 'quote_currency' or 'asset', which side of the ticker to use
        ticker: the ticker that should be used for the clock
        offset: volume offset for the calculation (float)

        :param kwargs
        chunk_size - how much trades data to get at once (we page to stop large memory footprint)
        """
        storage_period = self.recommended_storage_period(params)
        storage = MongoInfiniteTimeseriesStorage(self, storage_period,
                                                 mongo_connection=runner.feature_conn())
        super(FixedVolumeClock, self).__init__(runner, params, storage=storage, storage_period=storage_period,
                                               **kwargs)

        if params['offset'] > params['volume']:
            raise ValueError('Offset {} should be below volume {} for {}'.format(params['offset'], params['volume'],
                                                                                 self.__class__.__name__))

    @classmethod
    def _default_parameters(cls, params, runner):
        ticker = params.get('ticker', 'BTCUSD.SPOT.BITF')

        return {'start': get_first_valid_datetime(ticker),
                'volume': 1000,
                'offset': 0,
                'units': CryptoCrossSide.QUOTE_CURRENCY,
                'ticker': ticker,
                'source': CryptoDataSource.ARCTIC}

    @staticmethod
    def extract_volume_steps(volume_series, offset, step):

        cumulative_volume = volume_series.cumsum() + offset
        mods = cumulative_volume.mod(step)
        multipliers = (cumulative_volume - mods) / step

        ticks = multipliers.diff()
        ticks.iloc[0] = multipliers.iloc[0]
        remainder = mods.iloc[-1]
        mods = mods[ticks > 0]
        ticks = ticks[ticks > 0]

        return pd.DataFrame({'n_ticked': ticks, 'remainder': mods}), remainder

    def _compute_recursion(self, previous_tick, start_dt, end_dt):
        # Ignore ticker
        self._logger.info('Computing clock for {} to {}'.format(start_dt, end_dt))
        trades_feature_request = [FeatureRequest('CryptoTrades', {'source': self._params['source'],
                                                                  'ticker': self._params['ticker'],
                                                                  'aggregate': True}
                                                 )]

        all_ticks = []

        if previous_tick is None:
            remainder = -self._params['offset']
            previous_time = self._params['start']
        else:
            remainder = previous_tick['remainder']
            previous_time = previous_tick.name

        # Get trades aggregated according to their timestamp
        trades_df = self._runner.get_merged_dataframes(trades_feature_request, start_dt, end_dt)

        if not trades_df.empty:
            if self._params['units'] == CryptoCrossSide.QUOTE_CURRENCY:
                volume = trades_df['price'] * trades_df['size']
            elif self._params['units'] == CryptoCrossSide.ASSET:
                volume = trades_df['size']
            else:
                raise ValueError('Unknown unit type {}'.format(self._params['units']))

            ticks, remainder = self.extract_volume_steps(volume, remainder, self._params['volume'])

            if not ticks.empty:
                ticks['previous'] = ticks.index.to_series(keep_tz=True).shift(1)
                ticks.ix[0, 'previous'] = previous_time
                all_ticks.append(ticks)

        if all_ticks:
            return pd.concat(all_ticks)
        else:
            return self.get_empty_feature()

    @classmethod
    def get_empty_feature(cls):
        return pd.DataFrame([], columns=['timestamp', 'previous', 'n_ticked', 'remainder']).set_index('timestamp')

    @classmethod
    def recommended_storage_period(cls, params):
        return dt.timedelta(hours=24)


if __name__ == '__main__':
    from sgmtradingcore.analytics.features.crypto.runner import CryptoFeatureRunner
    from datetime import datetime
    import pytz

    fr = FeatureRequest('FixedVolumeClock', {})
    runner_ = CryptoFeatureRunner()
    params = {'start': datetime(2018, 5, 30, tzinfo=pytz.UTC),
              'ticker': 'BTCUSD.PERP.BMEX',
              }
    fr = FeatureRequest('FixedVolumeClock', params)
    runner_ = CryptoFeatureRunner()

    dfs = runner_.get_merged_dataframes([fr], '2018-06-01', '2018-06-03', repopulate=True)
    print dfs
    # fr = FeatureRequest('FixedPeriodClock', {})
    # dfs2 = runner_.get_merged_dataframes([fr], '2018-06-12', '2018-06-13')
    # print dfs2
    # pass
