import datetime
import itertools

import numpy as np
import pandas as pd
import pytz

from sgmtradingcore.analytics.features.crypto.market_data import TSInput, CryptoDataSource
from sgmtradingcore.analytics.features.crypto.transform import RecursiveAggregationFeature, AggregationFeature
from sgmtradingcore.analytics.features.request import FeatureRequest
from sgmtradingcore.analytics.features.storage import MongoInfiniteTimeseriesStorage
from stratagemdataprocessing.crypto.enums import get_first_valid_datetime


class HeikinAshi(RecursiveAggregationFeature):
    REQUIRED_PARAMETERS = {'start_fresh_after_nans'}

    def __init__(self, runner, params, **kwargs):
        """
        Heikin Ashi Candles
        :param runner:
        :param params:
        _input: input feature request to perform the candle arithmetic on
        _columns: (list of length 1) columns to take from the input feature
        _clock: Clock feature request which gives the time domain of each candle
        _start_fresh_after_nans: if True then if we have a lack of market data then we restart the candle calculation
        from that point. If not, we remember the last non-NaN value and continue calculating based on that.
        :param kwargs:
        """
        storage_period = self.recommended_storage_period(runner, params)
        storage = MongoInfiniteTimeseriesStorage(self, storage_period,
                                                 mongo_connection=runner.feature_conn())
        super(HeikinAshi, self).__init__(runner, params, storage, storage_period, **kwargs)

    @classmethod
    def get_empty_feature(cls):
        return pd.DataFrame(columns=['close', 'open', 'high',
                                     'low', 'open_last_nNan', 'close_last_nNan'])

    @classmethod
    def _default_parameters(cls, params, runner):
        ticker = None
        if 'input' in params:
            input_params = params['input'].feature_params(runner)
            if 'ticker' in input_params:
                ticker = input_params

        ticker = ticker or 'BTCUSD.SPOT.BITS'

        return {'input': FeatureRequest('CryptoTrades', {'ts_name': TSInput.TRADE_PRICE, 'ticker': ticker}),
                'columns': [TSInput.TRADE_PRICE],
                'start': get_first_valid_datetime(ticker),
                'clock': FeatureRequest('FixedPeriodClock', {'frequency': '1H'}),
                'start_fresh_after_nans': False}

    def _compute_recursive_tick(self, clock_tick, previous_tick, data):
        column = self._columns[0]

        market_data = data[column].dropna()

        if len(market_data) == 0:
            # No market data and no previous tick
            return {'open': np.nan,
                    'close': np.nan,
                    'high': np.nan,
                    'low': np.nan,
                    'open_last_nNan': previous_tick['open_last_nNan'],
                    'close_last_nNaN': previous_tick['close_last_nNan']}

        market_open = market_data.loc[data.first_valid_index()].mean()
        market_close = market_data.loc[data.last_valid_index()].mean()
        market_high = market_data.max()
        market_low = market_data.min()

        if previous_tick is None:
            # This should only hold if it's the first one
            return {'open': market_open,
                    'close': market_close,
                    'high': market_high,
                    'low': market_low,
                    'open_last_nNan': market_open,
                    'close_last_nNan': market_close}

        previous_open = previous_tick['open']
        previous_close = previous_tick['close']

        if np.isnan(previous_open) and self._params['start_fresh_after_nans']:
            previous_open = market_open
        else:
            previous_open = previous_tick['open_last_nNan']

        if np.isnan(previous_close) and self._params['start_fresh_after_nans']:
            previous_close = market_close
        else:
            previous_close = previous_tick['close_last_nNan']

        ha_close = (market_open + market_close + market_high + market_low) / 4.
        ha_open = (previous_open + previous_close) / 2.
        ha_high = max(market_high, ha_close, ha_open)
        ha_low = min(market_low, ha_close, ha_open)

        ha_open_last_nNan = ha_open or previous_tick['open_last_nNan']
        ha_close_last_nNan = ha_close or previous_tick['close_last_nNan']

        return {'open': ha_open,
                'close': ha_close,
                'low': ha_low,
                'high': ha_high,
                'open_last_nNan': ha_open_last_nNan,
                'close_last_nNan': ha_close_last_nNan}


class OHLC(AggregationFeature):

    def __init__(self, runner, params, **kwargs):

        storage_period = self.recommended_storage_period(runner, params)
        storage = MongoInfiniteTimeseriesStorage(self, storage_period,
                                                 mongo_connection=runner.feature_conn())
        super(OHLC, self).__init__(runner, params, storage, storage_period, **kwargs)

    @classmethod
    def _check_parameters(cls, params):
        if len(params['columns']) > 1:
            raise ValueError('{} only supports one column'.format(cls.__name__))

    @classmethod
    def _default_parameters(cls, params, runner):
        ticker = None

        if 'input' in params:
            input_params = params['input'].feature_params(runner)
            if 'ticker' in input_params:
                ticker = input_params['ticker']

        ticker = ticker or 'BTCUSD.SPOT.BITS'

        return {'input': FeatureRequest('CryptoTrades', {'ts_name': TSInput.TRADE_PRICE, 'ticker': ticker}),
                'columns': [TSInput.TRADE_PRICE],
                'clock': FeatureRequest('FixedPeriodClock', {'frequency': '1H'})}

    def _compute_ticks(self, groups):
        ret = groups[self._params['columns'][0]].agg({'high': 'max',
                                                      'low': 'min',
                                                      'open': lambda x: x.loc[x.first_valid_index()].mean(),
                                                      'close': lambda x: x.loc[x.last_valid_index()].mean()})
        return ret

    @classmethod
    def get_empty_feature(cls):
        return pd.DataFrame(columns=['close', 'open', 'high', 'low'])


def populate_candles():
    import tqdm
    class_names = ['OHLC', 'HeikinAshi']
    tickers = ['BTCUSD.SPOT.BITS', 'BTCUSD.SPOT.BITF', 'BTCUSD.PERP.BMEX']
    frequencies = ['5T', '15T', '30T', '1H', '4H']
    from sgmtradingcore.analytics.features.crypto.runner import CryptoFeatureRunner
    runner = CryptoFeatureRunner()

    combos = itertools.product(class_names, tickers, frequencies)
    valid_dt_dict = {}
    for cls, ticker, freq in tqdm.tqdm(list(combos)):
        if ticker not in valid_dt_dict:
            valid_dt_dict[ticker] = get_first_valid_datetime(ticker)

        start_date = get_first_valid_datetime(ticker)
        end_date = datetime.datetime.now(tz=pytz.UTC)

        params = {'input': FeatureRequest('CryptoTrades', {'source': CryptoDataSource.ARCTIC,
                                                           'ticker': ticker}),
                  'columns': [TSInput.TRADE_PRICE],
                  'start': start_date,
                  'clock': FeatureRequest('FixedPeriodClock', {'frequency': freq})}

        fr = FeatureRequest(cls, params)
        runner.precompute([fr], start_date, end_date, n_jobs=1)


if __name__ == '__main__':
    populate_candles()

