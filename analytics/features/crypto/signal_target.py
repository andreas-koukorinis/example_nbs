import pytz
import datetime
import pandas as pd

from sgmtradingcore.analytics.features.crypto.market_data import TSInput, CryptoDataSource
from sgmtradingcore.analytics.features.request import FeatureRequest
from sgmtradingcore.analytics.features.crypto.candles import OHLC


class PriceMoveBinary(OHLC):
    """
    Feature that checks whether the price over the duration of a candle increased
    or decreased - subtle difference from traditional OHLC in that it uses the first
    timestamp of the bar/candle/clock in the index rather than the last one
    """

    def _compute_ticks(self, groups):
        ret = super(PriceMoveBinary, self)._compute_ticks(groups)
        ret.index = groups['previous'].first()
        return pd.DataFrame(ret['close'] > ret['open'], columns=['target'])


class PriceMoveRegression(OHLC):
    """
    Similar to PriceMoveBinary except it measures the value of the price movement
    over the bar
    """
    def _compute_ticks(self, groups):
        ret = super(PriceMoveRegression, self)._compute_ticks(groups)
        ret.index = groups['previous'].first()
        return pd.DataFrame(ret['close'] - ret['open'], columns=['target'])


if __name__ == '__main__':

    from sgmtradingcore.analytics.features.crypto.runner import CryptoFeatureRunner
    runner = CryptoFeatureRunner()
    start_date = datetime.datetime(2018, 6, 1, tzinfo=pytz.utc)
    end_date = start_date + datetime.timedelta(days=5)
    ticker = 'BTCUSD.PERP.BMEX'
    params = {'input': FeatureRequest('CryptoTrades', {'source': CryptoDataSource.ARCTIC,
                                                       'ticker': ticker}),
              'columns': [TSInput.TRADE_PRICE],
              'start': start_date,
              }

    fr = FeatureRequest('PriceMoveRegression', params)
    df = runner.get_merged_dataframes([fr], start_date, end_date, repopulate=True)




