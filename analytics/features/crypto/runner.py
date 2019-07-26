import logging

import mongomock
from mock import patch

from sgmtradingcore.analytics.features.crypto.moving_averages import SimpleMovingAverage
from sgmtradingcore.analytics.features.crypto.candles import OHLC, HeikinAshi
from sgmtradingcore.analytics.features.crypto.signal_target import PriceMoveBinary, \
    PriceMoveRegression
from sgmtradingcore.analytics.features.crypto.volatility_features import BollingerBands
from sgmtradingcore.analytics.features.crypto.market_data import CryptoTrades, CryptoLOB
from sgmtradingcore.analytics.features.crypto.prices import BidAskPrice
from sgmtradingcore.analytics.features.crypto.clocks import FixedPeriodClock, FixedVolumeClock
from sgmtradingcore.analytics.features.crypto.volume import VolumeTraded
from sgmtradingcore.analytics.features.feature import InfiniteTimeSeriesFeature, merge_timeseries_features
from sgmtradingcore.analytics.features.runner import BaseFeatureRunner
from stratagemdataprocessing.crypto.market.arctic.arctic_storage import ArcticStorage
from stratagemdataprocessing.crypto.market.trades_cache import TradesCache
from stratagemdataprocessing.dbutils.mongo import MongoPersister
from stratagemdataprocessing.util.dateutils import parse_date_if_necessary

CRYPTO_FEATURES = [OHLC,
                   HeikinAshi,
                   BidAskPrice,
                   SimpleMovingAverage,
                   BollingerBands,
                   CryptoTrades,
                   CryptoLOB,
                   FixedPeriodClock,
                   FixedVolumeClock,
                   VolumeTraded,

                   # Signal targets
                   PriceMoveBinary,
                   PriceMoveRegression,
                   ]


class CryptoFeatureRunner(BaseFeatureRunner):
    def __init__(self, env='prod', **kwargs):
        """
        :param kwargs:
            trades_cache: TradesCache
        """

        extra_feature_classes = kwargs.get('extra_feature_classes', [])
        extra_feature_classes.extend(CRYPTO_FEATURES)

        _ = kwargs.pop('extra_feature_classes', None)
        self._env = env
        super(CryptoFeatureRunner, self).__init__(extra_feature_classes=extra_feature_classes, **kwargs)

    def _init_shared_objs(self):

        trades_cache = self._kwargs['trades_cache'] if 'trades_cache' in self._kwargs else TradesCache()

        if self._env.lower() == 'prod':
            feature_conn = MongoPersister.init_from_config('features', auto_connect=True)
        elif self._env.lower() == 'dev':
            feature_conn = MongoPersister.init_from_config('features_dev', auto_connect=True)
        elif self._env.lower() == 'mock':
            with patch('stratagemdataprocessing.dbutils.mongo.MongoClient', mongomock.MongoClient):
                feature_conn = MongoPersister.init_from_config('features', auto_connect=True)
        else:
            raise ValueError('Unknown Environment {}'.format(self._env))

        if 'arctic_storage' in self._kwargs:
            arctic_loader = self._kwargs['arctic_storage']
        else:
            if self._env.lower() == 'dev':
                arctic_db_conn = MongoPersister.init_from_config('arctic_crypto_dev', auto_connect=True)
            else:
                arctic_db_conn = MongoPersister.init_from_config('arctic_crypto', auto_connect=True)
            arctic_loader = ArcticStorage(arctic_db_conn.client)

        self.SHARED_OBJECTS = {
            'trades_cache': trades_cache,
            'feature_conn': feature_conn,
            'arctic_loader': arctic_loader
        }

    def cleanup(self):
        pass

    def get_merged_dataframes(self, feature_requests, start_dt, end_dt, repopulate=False):
        """
        Return a TimeseriesFeature by sticker for start_dt <= t < end_dt

        :param feature_requests:
        :return: {sticker: DataFrame} where Dataframe has all the features merged
        """

        start_dt = parse_date_if_necessary(start_dt, to_utc=True)
        end_dt = parse_date_if_necessary(end_dt, to_utc=True)

        if not isinstance(feature_requests, list):
            feature_requests = [feature_requests]

        self._instantiate_features(feature_requests)

        for request in feature_requests:
            feature_id = request.feature_id
            self.feature_map[feature_id].initialize(start_dt, end_dt)

        features_df = list()
        for request in feature_requests:
            feature_id = request.feature_id
            feature = self.feature_map[feature_id]

            if not isinstance(feature, InfiniteTimeSeriesFeature):
                raise ValueError("Not a InfiniteTimeSeriesFeature")

            logging.info("Requesting {} {} {}".format(feature_id, start_dt, end_dt))
            df = feature.get_df(start_dt, end_dt, repopulate=repopulate)
            df.columns = [request.prefix + c for c in df.columns]
            features_df.append(df)
        merged = merge_timeseries_features(features_df)
        return merged

    def precompute(self, feature_requests, start_dt, end_dt, n_jobs=None):
        if any([req.get_feature_class(self).RECURSIVE for req in feature_requests]):
            raise ValueError("Do not use multiprocessing with a recursive feature")

        self._precompute_infinite_timeseries(feature_requests, start_dt, end_dt, n_jobs=n_jobs, input_type='tickers')

    def compute_dataframes(self, feature_requests, start_dt, end_dt):
        """
        Compute but do not load into memory.
        """
        start_dt = parse_date_if_necessary(start_dt, to_utc=True)
        end_dt = parse_date_if_necessary(end_dt, to_utc=True)
        if not isinstance(feature_requests, list):
            feature_requests = [feature_requests]

        self._instantiate_features(feature_requests)

        for request in feature_requests:
            feature_id = request.feature_id
            self.feature_map[feature_id].initialize(start_dt, end_dt)

        for request in feature_requests:
            feature_id = request.feature_id
            feature = self.feature_map[feature_id]
            if not isinstance(feature, InfiniteTimeSeriesFeature):
                raise ValueError("Not a InfiniteTimeSeriesFeature")

            logging.info("Requesting {}, {} {}".format(feature_id, start_dt, end_dt))
            feature.compute(start_dt, end_dt)
        return

    def delete_features(self, feature_requests, start_dt, end_dt):
        """
        Delete the features from the caches
        """
        self._instantiate_features(feature_requests)
        for request in feature_requests:
            feature_id = request.feature_id
            logging.info("Deleting {}".format(feature_id))
            self.feature_map[feature_id].delete_range(start_dt, end_dt)


if __name__ == '__main__':
    from sgmtradingcore.analytics.features.request import FeatureRequest
    from stratagemdataprocessing.crypto.enums import generate_crypto_ticker, CryptoMarkets, CryptoExchange
    import datetime as dt
    from sgmtradingcore.analytics.features.crypto.market_data import TSInput
    import pytz

    start_dt = dt.datetime(2014, 3, 17, 0, 0, tzinfo=pytz.UTC)
    end_dt = dt.datetime(2014, 3, 20, 0, 0, tzinfo=pytz.UTC)

    requests = [
        FeatureRequest('HeikinAshiCandles',
                       {
                           'frequency': '15T',
                           'data_col_name': TSInput.L1_MID,
                       }
                       )

    ]

    # requests = [
    #     FeatureRequest('OHLC',
    #                    {
    #                        'frequency': '15m',
    #                        'data_col_name': 'l1_ask',
    #                        'input_data_cfg':
    #                            {
    #                                'is_feature_based': True,
    #                                'base_feature_name': 'BidAskPrice',
    #                                'base_feature_params': {'frequency': '1s'},
    #                                'base_feature_prefix': '1s_',
    #
    #                            }
    #                    }
    #                    )
    #
    # ]

    runner = CryptoFeatureRunner()

    tickers = [generate_crypto_ticker(mkt=CryptoMarkets.SPOT, exchange=CryptoExchange.BITSTAMP)]

    res = runner.compute_dataframes(requests, start_dt, end_dt)
