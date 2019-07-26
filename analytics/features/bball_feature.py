import copy
import datetime
import itertools

import numpy as np
import pandas as pd

from sgmtradingcore.analytics.features.feature import EventFeature
from sgmtradingcore.analytics.features.market_indicators import \
    MarketFeature
from sgmtradingcore.analytics.features.odds import ABCBestOdds
from sgmtradingcore.analytics.features.storage import \
    MongoEventFeatureStorage
from stratagemdataprocessing.dbutils.mongo import MongoPersister
from stratagemdataprocessing.enums.markets import BasketballMarkets, \
    BasketballSelections, Markets
from stratagemdataprocessing.enums.odds import Bookmakers, Sports
from stratagemdataprocessing.parsing.common.stickers import \
    generate_sticker, MarketScopes


def frange(a, b, step):
    while a < b:
        yield a
        a += step


class BBallPreLineupOdds(EventFeature):
    REQUIRED_PARAMETERS = {'source', 'bookmakers', 'markets'}

    def __init__(self, runner, params, **kwargs):
        """
        Note, this would be much better expressing the stickers in terms of difference to the NH
        (the resulting dataframe is very sparse)
        :param runner:
        :param params: type dict with the values:
            'source': 'ABC' or 'OddsCache'
            'bookmakers: [Bookmakers.PINNACLE_SPORTS]
            'markets': [BasketballMarkets]
        :param kwargs:
        """

        storage = MongoEventFeatureStorage(self)
        super(BBallPreLineupOdds, self).__init__(runner, params, storage)
        self._fixture_cache = runner.shared_objects()['fixture_cache']
        self._source = self._params['source']
        if self._source not in ['ABC', 'OddsCache']:
            raise ValueError('Unknown source {}'.format(self._source))
        self._kwargs = kwargs

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'source': 'ABC'}

    @classmethod
    def get_feature_id(cls, feature_class_name, params, runner, feature_conn=None, force_new=False):
        """
        Makes feature_id independent of the bms and markets orders
        """
        par = copy.deepcopy(params)
        par['bookmakers'] = sorted(par['bookmakers'])
        par['markets'] = sorted(par['markets'])
        return super(BBallPreLineupOdds, cls).get_feature_id(feature_class_name, par, runner,
                                                             feature_conn=feature_conn, force_new=force_new)

    def _compute_by_event_id(self, fixture_id, repopulate=False):
        params = {
            'bookmakers': self._params['bookmakers'],
            'markets': self._params['markets'],
        }
        if self._source == 'ABC':
            feature_request = FeatureRequest('BBallABCBestOdds', params, self._kwargs)
        elif self._source == 'OddsCache':
            feature_request = FeatureRequest('BBallOddsCacheBestOdds', params, self._kwargs)
        else:
            raise ValueError('Unknown source {}'.format(self._params['source']))

        market_data = self._runner.get_dataframes_by_event_ids([feature_request], [fixture_id])[fixture_id]

        if market_data is None or market_data.empty:
            # I think market_data might be an empty df instead
            return {}

        pre_lineup_time = self._fixture_cache.get_kickoff(fixture_id) - datetime.timedelta(hours=1)

        market_data = market_data[market_data.index < pre_lineup_time]
        return market_data.mean().to_dict()


class BBallOddsCacheBestOdds(MarketFeature):
    REQUIRED_PARAMETERS = {'markets', 'bookmakers'}

    def __init__(self, runner, params, **kwargs):
        """
        Get market data from the odds cache
        :param runner:
        :param params: type dict, with the keys:
            'markets': [BasketballMarkets]
            'bookmakers': [Bookmakers.BETFAIR]
        :param kwargs:
            'prefetch': [optional] pre fetch odds cache; useful if you need to recompute for a huge number of stickers
        """

        storage = None
        super(BBallOddsCacheBestOdds, self).__init__(runner, params, storage)
        self._bookmakers = self._params['bookmakers']
        self._markets = self._params['markets']

        self._fixture_cache = self._runner.shared_objects()['fixture_cache']
        self._odds_cache = self._runner.shared_objects()['odds_cache']
        self._precache = kwargs.get('precache', False)

    def _get_stickers_for_match(self, fixture_id):
        stickers = []
        if BasketballMarkets.FULL_TIME_MONEYLINE in self._markets:
            # Change on competition in fixture_cache
            selections = [BasketballSelections.HOME_TEAM, BasketballSelections.AWAY_TEAM]

            for (selection, book) in itertools.product(selections, self._bookmakers):
                stickers.append(generate_sticker(Sports.BASKETBALL,
                                                 (MarketScopes.EVENT, fixture_id),
                                                 BasketballMarkets.FULL_TIME_MONEYLINE,
                                                 selection,
                                                 bookmaker=book))

        if BasketballMarkets.FULL_TIME_POINT_SPREAD in self._markets:
            pts_range = frange(-20.5, 20.5, 1.0)
            selections = [BasketballSelections.HOME_TEAM, BasketballSelections.AWAY_TEAM]
            for (selection, book, line) in itertools.product(selections, self._bookmakers, pts_range):
                stickers.append(generate_sticker(Sports.BASKETBALL,
                                                 (MarketScopes.EVENT, fixture_id),
                                                 BasketballMarkets.FULL_TIME_POINT_SPREAD,
                                                 selection, line,
                                                 bookmaker=book))

        if BasketballMarkets.FULL_TIME_TOTAL_POINTS in self._markets:
            pts_range = frange(130.5, 240.5, 1.0)
            selections = [BasketballSelections.OVER, BasketballSelections.UNDER]
            for (selection, book, line) in itertools.product(selections, self._bookmakers, pts_range):
                stickers.append(generate_sticker(Sports.BASKETBALL,
                                                 (MarketScopes.EVENT, fixture_id),
                                                 BasketballMarkets.FULL_TIME_TOTAL_POINTS,
                                                 selection, line,
                                                 bookmaker=book))
        return stickers

    def _compute_by_event_id(self, fixture_id, repopulate=False):

        stickers = self._get_stickers_for_match(fixture_id)

        dfs = []
        for s in stickers:
            data = self._odds_cache.get(s)

            sticker_data = []
            sticker_index = []
            if data is None:
                continue

            for tick in data:
                sticker_index.append(tick.timestamp)
                odds = {s + '_back': tick.back[0].o if len(tick.back) else np.nan,
                        s + '_lay': tick.lay[0].o if len(tick.lay) else np.nan}
                sticker_data.append(odds)

            dfs.append(pd.DataFrame(index=sticker_index, data=sticker_data))

        return self._runner.merge_timeseries_features(dfs)


class BBallABCBestOdds(ABCBestOdds):
    @property
    def default_abc_connection(self):
        return MongoPersister.init_from_config(
            'abc', auto_connect=True)

    @property
    def sport_name(self):
        return 'basketball'

    @property
    def fixture_meta_col_name(self):
        return self.sport_name + '_fixture_meta'


if __name__ == "__main__":
    from sgmtradingcore.analytics.features.runner import FeatureRunner
    from sgmtradingcore.analytics.features.request import FeatureRequest
    import logging

    logging.basicConfig(level=logging.INFO)

    start_date = datetime.datetime(2017, 01, 29)
    end_date = datetime.datetime(2017, 02, 01)

    fixtures = [2241979]  # [f['gsm_id'] for f in find_football_fixtures(
    #     [8],start_date, end_date)]
    feature_requests = [FeatureRequest('FballABCBestOdds',
                                       {'bookmakers': [Bookmakers.BETFAIR],
                                        'markets': [Markets.FULL_TIME_ASIAN_HANDICAP_GOALS,
                                                    Markets.FULL_TIME_1X2,
                                                    Markets.FULL_TIME_OVER_UNDER_GOALS]},
                                       {}
                                       )]

    runner_ = FeatureRunner()

    fixture_data = runner_.get_dataframes_by_event_ids(
        feature_requests, fixtures)
    print fixture_data[2241979]
    fixture_data = runner_.get_event_features(feature_requests, fixtures)
    # import pprint
    #
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(fixture_data)
