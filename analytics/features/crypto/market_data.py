import numpy as np
import pandas as pd

from sgmtradingcore.analytics.features.feature import InfiniteTimeSeriesFeature
from sgmtradingcore.analytics.features.storage import EmptyFeatureStorage
from sgmtradingcore.util.misc import daterange
from stratagemdataprocessing.crypto.market.trades import Status


class TSInput(object):
    L1_BID = 'BIDPRC1'
    L1_ASK = 'ASKPRC1'
    L1_SPREAD = 'SPREAD1'
    L1_MID = 'MID1'
    L1_MICRO = 'MICROPRICE1'
    L1_PRICE = 'PRCS1'
    TRADE_PRICE = 'price'

    # call back func to get data from flat files

    @staticmethod
    def l1_bid_from_tick(tick):
        try:
            return {'timestamp': tick.received_at, TSInput.L1_BID: tick.bids[0].p}
        except IndexError:
            return {'timestamp': tick.received_at, TSInput.L1_BID: np.nan}

    @staticmethod
    def l1_ask_from_tick(tick):
        try:
            return {'timestamp': tick.received_at, TSInput.L1_ASK: tick.asks[0].p}
        except IndexError:
            return {'timestamp': tick.received_at, TSInput.L1_ASK: np.nan}

    @staticmethod
    def l1_spread_from_tick(tick):
        try:
            return {'timestamp': tick.received_at,
                    TSInput.L1_SPREAD: tick.asks[0].p - tick.bids[0].p}
        except IndexError:
            return {'timestamp': tick.received_at, TSInput.L1_SPREAD: np.nan}

    @staticmethod
    def l1_mid_from_tick(tick):
        try:
            return {'timestamp': tick.received_at,
                    TSInput.L1_MID: (tick.asks[0].p + tick.bids[0].p) / 2.}
        except IndexError:
            return {'timestamp': tick.received_at, TSInput.L1_MID: np.nan}

    @staticmethod
    def l1_micro_from_tick(tick):
        try:
            return {'timestamp': tick.received_at,
                    TSInput.L1_MICRO: float(tick.bids[0].v * tick.asks[0].p +
                                       tick.asks[0].v * tick.bids[0].p) / (
                                          tick.asks[0].v + tick.bids[0].v)}
        except IndexError:
            return {'timestamp': tick.received_at, TSInput.L1_MICRO: np.nan}

    @staticmethod
    def l1_prices(tick):
        bid_price = tick.bids[0].p if len(tick.bids) > 0 else np.nan
        ask_price = tick.asks[0].p if len(tick.asks) > 0 else np.nan

        return {
            'timestamp': tick.received_at,
            TSInput.L1_BID: bid_price,
            TSInput.L1_ASK: ask_price
        }


DATA_INPUT_CALLBACKS = {
    TSInput.L1_ASK: TSInput.l1_ask_from_tick,
    TSInput.L1_BID: TSInput.l1_bid_from_tick,
    TSInput.L1_SPREAD: TSInput.l1_spread_from_tick,
    TSInput.L1_MID: TSInput.l1_mid_from_tick,
    TSInput.L1_MICRO: TSInput.l1_micro_from_tick,
    TSInput.L1_PRICE: TSInput.l1_prices,
}


class CryptoDataSource(object):
    TSDB = 'tsdb'
    RAW_FILES = 'raw_files'
    ARCTIC = 'arctic'


class CryptoTrades(InfiniteTimeSeriesFeature):
    REQUIRED_PARAMETERS = {'source', 'ticker', 'aggregate'}

    def __init__(self, runner, params, **kwargs):
        """
        Feature to retrieve historical trades
        :param runner:
        :param params:
        source: CryptoDataSource, where we retrieve the trade data
        ticker: string, which instrument
        aggregate: boolean, whether we aggregate trades according to their timestamp i.e. one
        trade is likely to be composed of multiple orders with identical timestamps, we will
        combine them into a single trade
        :param kwargs:
        :return:
        """
        storage = EmptyFeatureStorage(self)
        self._ticker = params['ticker']
        super(CryptoTrades, self).__init__(runner, params, storage=storage, storage_period=None,
                                           **kwargs)
        self._aggregate = self._params['aggregate']

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'source': CryptoDataSource.ARCTIC,
                'ticker': 'BTCUSD.SPOT.BITS',
                'aggregate': False}

    def _compute(self, start_dt, end_dt):
        if self._params['source'] == CryptoDataSource.ARCTIC:
            data_for_range = self._runner.SHARED_OBJECTS['arctic_loader'].load_trades(self._ticker,
                                                                                      start_dt,
                                                                                      end_dt)

        elif self._params['source'] == CryptoDataSource.RAW_FILES:
            cache = self._runner.SHARED_OBJECTS['trades_cache']
            data_for_range = []
            for day in daterange(start_dt.date(), end_dt.date()):
                data_for_range.extend(cache.get_trades(self._ticker, day))

        else:
            raise ValueError('Unknown Source')

        result_df = pd.DataFrame(d.to_dict() for d in data_for_range)
        if not result_df.empty:
            result_df.set_index('happened_at', inplace=True)
            if self._aggregate:
                return self._aggregate_trades(result_df)

        return result_df

    @staticmethod
    def _aggregate_trades(trades_df):
        df = trades_df.copy()
        df['sp'] = df['size'] * df['price']
        df2 = df.groupby(df.index).agg({'sp': 'sum',
                                        'size': 'sum',
                                        'price': {'high': 'max',
                                                  'low': 'min',
                                                  'n': 'size'
                                                  },
                                        'side': {'first': 'first',
                                                 'n': 'unique'},
                                        })

        df3 = pd.DataFrame([], index=df2.index)
        df3['price'] = df2[('sp', 'sum')] / df2[('size', 'sum')]
        df3['size'] = df2[('size', 'sum')]
        df3['n'] = df2[('price', 'n')]
        df3['high'] = df2[('price', 'high')]
        df3['low'] = df2[('price', 'low')]
        df3['side'] = df2[('side', 'first')]
        df3['n_sides'] = df2[('side', 'n')].apply(len)

        return df3


class CryptoLOB(InfiniteTimeSeriesFeature):
    REQUIRED_PARAMETERS = {'source', 'ts_name', 'ticker', 'statuses'}

    def __init__(self, runner, params, **kwargs):
        storage = EmptyFeatureStorage(self)
        self._ticker = params['ticker']
        super(CryptoLOB, self).__init__(runner, params, storage=storage, storage_period=None,
                                        **kwargs)
        self._transform_function = DATA_INPUT_CALLBACKS[self._params['ts_name']]

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'ts_name': TSInput.L1_MID,
                'source': CryptoDataSource.ARCTIC,
                'statuses': [Status.OK]}

    def _compute(self, start_dt, end_dt):
        if self._params['source'] == CryptoDataSource.ARCTIC:
            data_for_range = self._runner.SHARED_OBJECTS['arctic_loader'].load_lob(self._ticker,
                                                                                   start_dt, end_dt,
                                                                                   status_list=
                                                                                   self._params[
                                                                                       'statuses'])

        elif self._params['source'] == CryptoDataSource.RAW_FILES:

            cache = self._runner.SHARED_OBJECTS['trades_cache']
            data_for_range = []
            for day in daterange(start_dt.date(), end_dt.date()):
                data_for_range.extend(cache.get_lob(self._ticker, day))
        else:
            raise ValueError('Unknown source {}'.format(self._params['source']))

        result_df = pd.DataFrame([self._transform_function(tick) for tick in data_for_range])
        if not result_df.empty:
            result_df.set_index('timestamp', inplace=True)

        return result_df


if __name__ == '__main__':
    from sgmtradingcore.analytics.features.request import FeatureRequest
    from sgmtradingcore.analytics.features.crypto.runner import CryptoFeatureRunner
    import datetime
    from stratagemdataprocessing.crypto.enums import get_first_valid_datetime

    runner_ = CryptoFeatureRunner()

    tickers_ = ['BTCUSD.SPOT.BITF']
    start_dt_ = get_first_valid_datetime()[tickers_[0]]
    end_dt_ = start_dt_ + datetime.timedelta(days=2)  # datetime.datetime(2014, 6, 30, 0, 0, tzinfo=pytz.UTC)

    requests = [FeatureRequest('CryptoTrades',
                               {'source': CryptoDataSource.RAW_FILES,
                                'ticker': tickers_[0],
                                'aggregate': True
                                })]
    cache_trades = runner_.get_merged_dataframes(requests, start_dt_, end_dt_)

    requests = [FeatureRequest('CryptoTrades', {'source': CryptoDataSource.ARCTIC,
                                                'ticker': tickers_[0]})]
    arctic_trades = runner_.get_merged_dataframes(requests, start_dt_, end_dt_)

    requests = [FeatureRequest('CryptoLOB', {'source': CryptoDataSource.RAW_FILES,
                                             'ticker': tickers_[0]})]
    cache_lob = runner_.get_merged_dataframes(requests, start_dt_, end_dt_)

    requests = [FeatureRequest('CryptoLOB', {'source': CryptoDataSource.ARCTIC,
                                             'ticker': tickers_[0]})]
    arctic_lob = runner_.get_merged_dataframes(requests, start_dt_, end_dt_)
    pass
