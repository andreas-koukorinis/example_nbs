from sgmtradingcore.analytics.features.crypto.transform import TransformationFeature
from sgmtradingcore.analytics.features.crypto.market_data import TSInput, CryptoDataSource
from sgmtradingcore.analytics.features.request import FeatureRequest

import pandas as pd
from pandas.tseries.frequencies import to_offset


class BollingerBands(TransformationFeature):
    """
    Feature implementation of Bollinger bands, a time series indicator for the volatility around a moving average.
    Typically a trading strategy would be looking for the moving average of a time series within a tube defined
    by its rolling vol. Some details here:
     https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_bands

     Note: Lock down the input provider to be OHLC only, fit with is described in trading docs.
            The implementation here relies on this assumption.
     Attributes:
        ------------------------------------
        _window (int): Lookback window for the calculation of rolling mean and vol
    """
    REQUIRED_PARAMETERS = {'window'}

    def __init__(self, runner, params, **kwargs):
        super(BollingerBands, self).__init__(runner, params, **kwargs)
        if self._input.feature_class_name is not 'OHLC':
            raise ValueError("Bollinger bands only work with OHLC input.")

        self._window = params['window']

    @classmethod
    def _default_parameters(cls, params, runner):

        lob_input = FeatureRequest('CryptoLOB', {'ts_name': TSInput.L1_MID,
                                                 'ticker': 'BTCUSD.SPOT.BITF',
                                                 'source': CryptoDataSource.ARCTIC})

        clock = FeatureRequest('FixedPeriodClock', {'frequency': '15T'})

        return {'input': FeatureRequest('OHLC', {'input': lob_input, 'columns': [TSInput.L1_MID],
                                                 'clock': clock}),
                'columns': ['close']}

    def _compute(self, start_dt, end_dt):

        input_data_frequency = self._input.feature_params(self._runner)['frequency']

        window_offset_for_first_point = pd.to_timedelta(to_offset(input_data_frequency))
        # We need to move start time window units of frequency back, to make sure the first point can be computed:
        updated_start = start_dt - self._window * window_offset_for_first_point

        ohlc_data = self._input_feature(updated_start, end_dt)
        if len(ohlc_data) == 0:
            return pd.DataFrame(columns=['middle_band', 'lower_band', 'upper_band'])

        data_column = self._columns[0]
        ohlc_data['middle_band'] = ohlc_data[data_column].rolling(self._window, min_periods=self._window).mean()
        ohlc_close_std = ohlc_data[data_column].rolling(self._window, min_periods=self._window).std()
        ohlc_data['upper_band'] = ohlc_data['middle_band'] + 2 * ohlc_close_std
        ohlc_data['lower_band'] = ohlc_data['middle_band'] - 2 * ohlc_close_std

        # now drop whatever you used to facilitate computation:
        ohlc_data = ohlc_data[ohlc_data.index >= start_dt]
        return ohlc_data[['middle_band', 'lower_band', 'upper_band']]


if __name__ == '__main__':
    from sgmtradingcore.analytics.features.crypto.runner import CryptoFeatureRunner
    import pytz
    import datetime as dt

    # first valid timestamp BITS 17/3/2014
    tickers = ['BTCUSD.SPOT.BITS']

    runner = CryptoFeatureRunner(env='dev')

    start_dt = dt.datetime(2018, 1, 1, 0, 0, tzinfo=pytz.UTC)
    end_dt = dt.datetime(2018, 2, 28, 0, 0, tzinfo=pytz.UTC)

    input0 = FeatureRequest('CryptoLOB', {'ts_name': TSInput.L1_MID,
                                          'ticker': 'BTCUSD.SPOT.BITS'})

    inputRequest = FeatureRequest('OHLC', {'frequency': '15T', 'input': input0})

    requests = [
        FeatureRequest('BollingerBands', {'window': 20, 'input': inputRequest, 'columns': ['close']})
    ]

    res = runner.get_merged_dataframes(requests, start_dt, end_dt, repopulate=True)
