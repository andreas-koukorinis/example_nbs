from sgmtradingcore.analytics.features.crypto.candle_provider import simple_moving_average
from sgmtradingcore.analytics.features.crypto.transform import TransformationFeature
from pandas.tseries.frequencies import to_offset
import pandas as pd


class SimpleMovingAverage(TransformationFeature):
    # Return MA column containing the simple moving average (fixed frequency)

    REQUIRED_PARAMETERS = {'window', 'frequency'}

    def __init__(self, runner, params, **kwargs):
        """

        :param runner: CryptoMarketRunner
        :param params: type dict:
            'input': FeatureRequest for input feature
            'columns': columns (list of str) to do the moving average on
            'window' : rolling moving average period (string for pandas resampling.)
        :param kwargs:
            None
        """
        self._window = params['window']
        super(SimpleMovingAverage, self).__init__(runner, params, **kwargs)

    @property
    def frequency(self):
        return pd.Timedelta(to_offset(self._params['frequency'])).to_pytimedelta()

    def _compute(self, start_dt, end_dt):
        prev_dt = start_dt - self._params['window'] * self.frequency

        input_data = self._input_feature(prev_dt, end_dt)[self._params['columns']]

        out = simple_moving_average(input_data, window=self._params['window'],
                                    freq=self._params['frequency'], col=self._params['columns'])
        if len(out):
            out = out[out.index >= start_dt]
        out.index.name = self.INDEX_FIELD
        out.sort_index()
        return out
