import datetime as dt

from sgmtradingcore.analytics.features.crypto.transform import AggregationFeature
from sgmtradingcore.analytics.features.crypto.market_data import TSInput, CryptoDataSource
from sgmtradingcore.analytics.features.request import FeatureRequest


class BidAskPrice(AggregationFeature):
    """
    Tsdb feature for bid and ask prices for a given crypto ticker (= cross + exchange + market). Allow
    a down sampling of frequency for practical purposes. Now averaging across the period, just downsampling
    Attributes:
        _period (timedelta): Control the sapling frequency
    """

    def __init__(self, runner, params, **kwargs):
        super(BidAskPrice, self).__init__(runner, params, **kwargs)

    @classmethod
    def _default_parameters(cls, params, runner):
        # This feature is for L1 prices only so the configuration of the input data component is fixed here:
        return {
            'input': FeatureRequest('CryptoLOB', {'ts_name': TSInput.L1_PRICE,
                                                  'ticker': 'BTCUSD.SPOT.BITF',
                                                  'source': CryptoDataSource.ARCTIC}),
            'columns': ['BIDPRC1', 'ASKPRC1'],
            'clock': FeatureRequest('FixedPeriodClock', {'frequency': '1S'})
        }

    def _compute_ticks(self, groups):
        return groups.last()


if __name__ == '__main__':
    from sgmtradingcore.analytics.features.crypto.runner import CryptoFeatureRunner
    import pytz

    runner = CryptoFeatureRunner(env='dev')

    start_dt = dt.datetime(2018, 1, 1, 0, 0, tzinfo=pytz.UTC)
    end_dt = dt.datetime(2018, 1, 4, 0, 0, tzinfo=pytz.UTC)

    requests = [FeatureRequest('BidAskPrice', {}, prefix='1s_')]

    res = runner.get_merged_dataframes(requests, start_dt, end_dt)
    # runner.delete_features(requests, tickers, start_dt, end_dt)
    pass
