import pandas as pd

from sgmtradingcore.analytics.features.crypto.market_data import TSInput, CryptoDataSource
from sgmtradingcore.analytics.features.crypto.transform import AggregationFeature
from sgmtradingcore.analytics.features.storage import MongoInfiniteTimeseriesStorage
from stratagemdataprocessing.crypto.enums import CryptoCrossSide
from sgmtradingcore.analytics.features.request import FeatureRequest


class VolumeTraded(AggregationFeature):
    """
    Calculate an index of the daily volume, obviously derived from the trading data.
    """

    def __init__(self, runner, params, **kwargs):
        storage_period = self.recommended_storage_period(runner, params)
        storage = MongoInfiniteTimeseriesStorage(self, storage_period,
                                                 mongo_connection=runner.feature_conn())
        super(VolumeTraded, self).__init__(runner, params, storage=storage,
                                           storage_period=storage_period, **kwargs)

    @classmethod
    def _check_parameters(cls, params):
        if params['input'].feature_class_name is not 'CryptoTrades':
            raise ValueError("{} requires input from trades data only".format(cls.__name__))

    @classmethod
    def _default_parameters(cls, params, runner):
        if 'input' in params:
            ticker = params['input'].feature_params(runner)['ticker']
        else:
            ticker = 'BTCUSD.SPOT.BITS'

        return {'input': FeatureRequest('CryptoTrades',
                                        {'source': CryptoDataSource.ARCTIC,
                                         'ticker': ticker}),
                'clock': FeatureRequest('FixedPeriodClock', {'frequency': '1S'}),
                'columns': ['size']}

    def _compute_ticks(self, groups):
        return groups.agg({'size': 'sum'}).rename(columns={'size': 'traded_volume'})

    @classmethod
    def get_empty_feature(cls):
        return pd.DataFrame(columns=['traded_volume'])


if __name__ == '__main__':
    from sgmtradingcore.analytics.features.crypto.runner import CryptoFeatureRunner
    import datetime
    import pytz

    runner_ = CryptoFeatureRunner()
    ticker = 'BTCUSD.PERP.BMEX'
    clock = FeatureRequest('FixedVolumeClock', {'start': datetime.datetime(2018, 5, 30, tzinfo=pytz.UTC),
                                                'volume': 100000,
                                                'offset': 0,
                                                'units': CryptoCrossSide.ASSET,
                                                'ticker': ticker,
                                                'source': CryptoDataSource.ARCTIC})

    input = FeatureRequest('CryptoTrades', {'source': CryptoDataSource.ARCTIC,
                                            'ticker': 'BTCUSD.PERP.BMEX',
                                            'aggregate': True})

    fr = FeatureRequest('VolumeTraded', {'clock': clock,
                                         'input': input,
                                         }
                        )

    df = runner_.get_merged_dataframes(fr, '2018-06-01', '2018-06-03', repopulate=True)
    print df
    pass
