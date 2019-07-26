from collections import deque

from sgmtradingcore.analytics.features.feature import TimeSeriesFeature
from sgmtradingcore.analytics.features.market_indicators import MarketFeature
from sgmtradingcore.analytics.features.storage import EmptyFeatureStorage, InMemoryFeatureStorage, \
    PickleFileFeatureStorage
import pandas as pd

from stratagemdataprocessing.enums.markets import TennisMarkets, BasketballMarkets, TennisSelections, \
    BasketballSelections
from stratagemdataprocessing.enums.odds import Bookmakers, Sports
from stratagemdataprocessing.events.sports import get_sport_from_event_id
from stratagemdataprocessing.parsing.common.stickers import generate_sticker, MarketScopes


class VPINBasicFeature(MarketFeature):

    REQUIRED_PARAMETERS = {'bucket_size', 'bucket_number'}

    def __init__(self, runner, params, **kwargs):
        storage = InMemoryFeatureStorage(self, parent_cache=PickleFileFeatureStorage(self))
        super(VPINBasicFeature, self).__init__(runner, params, storage, **kwargs)
        self.bucket_size = self._params['bucket_size']
        self.bucket_number = self._params['bucket_number']

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'bucket_size': 1000,
                'bucket_number': 50}

    def _compute_by_sticker(self, sticker):

        requests = [FeatureRequest('VolumeBucketFeature',
                                                 {'bookmakers': self._bookmakers, 'bucket_size': self.bucket_size}, {})]
        buckets = list(self._runner.get_dataframes_by_stickers(requests, [sticker])[sticker].itertuples())

        total_vol = self.bucket_size * self.bucket_number
        bucket_q = deque(buckets[:self.bucket_number], maxlen=self.bucket_number)
        times = []
        vpin = sum(abs(i.back - i.lay) for i in bucket_q)
        vpins = [vpin / total_vol]
        if len(buckets) < self.bucket_number:
            return pd.DataFrame([], columns=['timestamp', 'vpin']).set_index('timestamp')

        times.append(buckets[self.bucket_number - 1].Index)

        for bucket in buckets[self.bucket_number:]:
            back = bucket.back
            lay = bucket.lay
            time = bucket.Index

            if back + lay < self.bucket_size - 0.01:
                print 'Underfull bucket'
                continue
            remove = bucket_q.popleft()
            bucket_q.append(bucket)

            vpin += abs(back - lay) - abs(remove.back - remove.lay)
            vpins.append(vpin / total_vol)
            times.append(time)

        return pd.DataFrame(zip(times, vpins), columns=['timestamp', 'vpin']).set_index('timestamp')


class VPINBasicTSFeature(TimeSeriesFeature):

    REQUIRED_PARAMETERS = {'bucket_size', 'bucket_number'}

    def _compute_by_sticker(self, sticker):
        pass

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'bucket_size': 1000,
                'bucket_number': 50}

    def __init__(self, runner, params, **kwargs):
        storage = EmptyFeatureStorage(self)
        super(VPINBasicTSFeature, self).__init__(runner, params, storage, **kwargs)
        self.bucket_size = self._params['bucket_size']
        self.bucket_number = self._params['bucket_number']

    def _compute_by_event_id(self, event_id, repopulate=False):

        sport = get_sport_from_event_id(event_id, self._runner.shared_objects()['mysql_client'])
        if sport == Sports.TENNIS:
            market = TennisMarkets.MATCH_ODDS
            selection = TennisSelections.PLAYER_A
        elif sport == Sports.BASKETBALL:
            market = BasketballMarkets.FULL_TIME_MONEYLINE
            selection = BasketballSelections.HOME_TEAM
        else:
            raise ValueError('Unknown Sport')
        sticker = generate_sticker(sport, (MarketScopes.EVENT, event_id), market, selection,
                                   bookmaker=self._params['bookmakers'][0])

        request = [FeatureRequest('VPINBasicFeature', self._params, {})]

        return self._runner.get_dataframes_by_stickers(request, [sticker])[sticker]


if __name__ == '__main__':
    from sgmtradingcore.analytics.features.runner import FeatureRunner
    from sgmtradingcore.analytics.features.request import FeatureRequest

    # stickers = ['T-EENP2862146-FT12-A.BF']
    # runner = FeatureRunner()
    # requests = [FeatureRequest('VPINBasicFeature',
    #                            {'bookmakers': [Bookmakers.BETFAIR], 'bucket_size': 1000, 'bucket_number': 50})]
    # dfs = runner.get_dataframes_by_stickers(requests, stickers, repopulate=True)
    runner_ = FeatureRunner()
    troublesome_sticker = 'T-EENP2733289-FT12-A.BF'
    request_ = FeatureRequest('LobOrdersFeature', {'bookmakers': [Bookmakers.BETFAIR]})
    runner_.get_dataframes_by_stickers([request_], [troublesome_sticker])

    pass
