import pandas as pd
from sgmtradingcore.analytics.features.request import FeatureRequest

from sgmtradingcore.analytics.features.market_indicators import MarketFeature
from sgmtradingcore.analytics.features.storage import EmptyFeatureStorage
from stratagemdataprocessing.bookmakers.common.date_utilities import unixepoch2datetime


class VolumeBucketFeature(MarketFeature):

    REQUIRED_PARAMETERS = {'bucket_size'}

    def __init__(self, runner, params, **kwargs):

        storage = EmptyFeatureStorage(self)
        super(VolumeBucketFeature, self).__init__(runner, params, storage, **kwargs)
        self.bucket_size = self._params.get('bucket_size', 1000)

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'bucket_size': 1000}

    @staticmethod
    def calculate_effective_volume(row):
        if row['side'] == 'lay':
            return row['size'] * (row['price'] - 1.)
        elif row['side'] == 'back':
            return row['size']

    @staticmethod
    def _new_bucket():
        return {'back': 0.0, 'lay': 0.0, 'timestamp': 0}

    def _compute_by_sticker(self, sticker):

        requests = [FeatureRequest('LobOrdersFeature', self._params, {})]

        lob_orders = self._runner.get_dataframes_by_stickers(requests, [sticker])[sticker]
        if lob_orders.empty:
            return pd.DataFrame([])

        matched_orders = lob_orders[lob_orders['order_type'] == 'matched']

        matched_orders['volume'] = matched_orders.apply(self.calculate_effective_volume, axis=1)

        buckets = []

        time_field = 'timestamp'
        volume_field = 'volume'
        side_field = 'side'

        current_bucket = self._new_bucket()
        v_to_fill = self.bucket_size

        # Bucketing process
        for row in matched_orders.itertuples():
            time = getattr(row, time_field)
            volume = getattr(row, volume_field)
            side = getattr(row, side_field)

            while volume > v_to_fill:
                current_bucket[side] += v_to_fill
                current_bucket['timestamp'] = unixepoch2datetime(time, milliseconds=True)
                buckets.append(current_bucket)
                current_bucket = self._new_bucket()
                v_to_fill = self.bucket_size
                volume -= v_to_fill

            if volume > 0:
                current_bucket[side] += volume
                v_to_fill -= volume

        return pd.DataFrame(buckets).set_index('timestamp')