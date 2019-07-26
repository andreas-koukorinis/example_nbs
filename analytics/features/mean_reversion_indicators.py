import datetime

from sgmresearchbase.analysis.mean_reversion.indicators import IndicatorSet
from sgmtradingcore.analytics.features.market_indicators import MarketFeature
from sgmtradingcore.analytics.features.storage import PickleFileFeatureStorage
from stratagemdataprocessing.enums.odds import Bookmakers
from stratagemdataprocessing.parsing.common.stickers import sticker_parts_from_sticker, extract_tradeable, extract_event


class MeanReversionIndicators(MarketFeature):
    REQUIRED_PARAMETERS = {'source', 'indicator_spec', 'rolling_params', 'indicator_frequency'}
    def __init__(self, runner, params, **kwargs):
        """
        Applies Mean Reversion Indicator to sticker prices
        for two hours before the fixture to the end of the sticker file

        :param runner:
        :param params: type dict:
            indicator_spec: inputs to the IndicatorSet
            source:
        :param kwargs:
        """

        storage = PickleFileFeatureStorage(self)
        super(MeanReversionIndicators, self).__init__(
            runner, params, storage=storage, **kwargs)
        self._source = params['source']
        self._bookmakers = params['bookmakers']
        self._indicator_spec = params['indicator_spec']
        self._rolling_params = params['rolling_params']
        self._indicator_frequency = params['indicator_frequency']

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'bookmakers': [Bookmakers.BETFAIR, Bookmakers.PINNACLE_SPORTS],
                'indicator_spec': None,
                'rolling_params': None,
                'indicator_frequency': 15,
                'source': 'MicroPrice'}


    def _compute_by_sticker(self, sticker):
        if self._source == 'MicroPrice':
            odds_feature_request = FeatureRequest(
                'MicroPrice', {
                    'bookmakers': self._bookmakers,
                    'precache': True
                }, {})
            cols = ['MicroPrice']
        else:
            raise ValueError('Unknown Source {}'.format(self._source))

        fc = self._runner.shared_objects()['fixture_cache']
        start_time = fc.get_kickoff(str(extract_event(sticker))) - datetime.timedelta(hours=6)

        dfs = self._runner.get_dataframes_by_stickers([odds_feature_request],
                                                      [sticker])

        in_df = dfs[sticker]
        in_df = in_df[in_df.index > start_time].resample('{}s'.format(self._indicator_frequency)).pad()

        indicator_set = IndicatorSet(
            indicator_spec=self._indicator_spec,
            rolling_params=self._rolling_params)

        out_df = indicator_set._apply_indicators_rolling_fixture(in_df, cols)
        del out_df['MicroPrice']
        return out_df


if __name__ == "__main__":
    from runner import FeatureRunner
    from sgmtradingcore.analytics.features.request import FeatureRequest
    from sgmbasketball.data.common import get_fixtures_info
    from sgmbasketball.utils import game_id_to_string

    runner = FeatureRunner()
    request = FeatureRequest('MeanReversionIndicators', {}, {})

    events = get_fixtures_info(season=2018, comp='NBA')

    game_ids = [game_id_to_string(e) for e in events['fixture_id'].tolist()]

    sticker_request = FeatureRequest('AvailableStickers',
                                     {'bookmakers': [Bookmakers.PINNACLE_SPORTS, Bookmakers.BETFAIR]}, {})
    stickers = runner.get_event_features(sticker_request, game_ids)
    flat_sticker_parts = [sticker_parts_from_sticker(s) for events, av_stick in stickers.iteritems() for s in
                          av_stick['AvailableStickers']['stickers']]

    fr = [FeatureRequest('MeanReversionIndicators', {'bookmakers': [Bookmakers.BETFAIR], 'indicator_frequency': 5})]
    all_mean_reversion = runner.get_dataframes_by_stickers_multithread(fr, [extract_tradeable(s.sticker) for s in
                                                                            flat_sticker_parts], n_threads=50)
    pass

