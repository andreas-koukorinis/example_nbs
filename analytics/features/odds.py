import pandas as pd

from sgmtradingcore.analytics.features.feature import TimeSeriesFeature
from stratagemdataprocessing.parsing.common.stickers import sticker_parts_from_sticker


class ABCBestOdds(TimeSeriesFeature):

    REQUIRED_PARAMETERS = {'bookmakers', 'markets'}

    def __init__(self, runner, params, storage=None, **kwargs):
        """
        :param runner:
        :param params: type dict, with the keys:
            'markets': e.g. BasketballMarkets or Markets
            'bookmakers': [Bookmakers.BETFAIR]
        :param kwargs:
        """
        self.conn = kwargs.get('abc_connection', self.default_abc_connection)
        super(ABCBestOdds, self).__init__(runner, params, storage)

        self._bookmakers = self._params['bookmakers']
        self._markets = self._params['markets']

    @property
    def default_abc_connection(self):
        """
        The connection of the abc database, if not specified in kwargs
        """
        raise NotImplementedError('Should be implemented in subclass')

    @property
    def sport_name(self):
        """
        Should match the prefix in the abc database
        
        :return (str) the sport name
        """
        #
        raise NotImplementedError('Should be implemented in subclass')

    @property
    def fixture_meta_col_name(self):
        raise NotImplementedError('Should be implemented in subclass')

    def _get_stickers_for_match(self, fixture_id):
        """
        Parameters
        ----------
        fixture_id: (str) fixture_id, e.g. GSM1234567 or ENP1234567
        """
        all_stickers = self.conn.db[self.fixture_meta_col_name].find_one(
            {"eid": fixture_id})
        if all_stickers is None or 'stkrs' not in all_stickers:
            return []

        ret = []
        for s in all_stickers['stkrs']:
            p = sticker_parts_from_sticker(s)
            if p.market in self._markets and p.bookmaker in self._bookmakers:
                ret.append(s)
        return ret

    def _compute_by_event_id(self, fixture_id, repopulate=False):
        """
        Parameters
        ----------
        fixture_id: (str) fixture_id, e.g. GSM1234567 or ENP1234567
        """
        stickers = self._get_stickers_for_match(fixture_id)

        if not stickers:
            return pd.DataFrame(index=pd.Index([], name='timestamp'))

        dfs_sticker = []
        for sticker in stickers:
            dfs_ticks = []
            data = self.conn.db[self.sport_name + '_market_data'].find(
                {'stkr.str': sticker,
                 'stkr.eid': fixture_id})

            for d in data:
                sticker_data = []
                sticker_index = []

                sticker = d['stkr']['str']

                for tick in d['tcks']:
                    sticker_index.append(tick['t'])
                    sticker_data.append({sticker + '_back': tick['bp'],
                                         sticker + '_lay': tick['lp']})

                dfs_ticks.append(pd.DataFrame(data=sticker_data, index=sticker_index))
            dfs_ticks = pd.concat(dfs_ticks, axis=0)
            dfs_sticker.append(dfs_ticks)
        df_odds = self._runner.merge_timeseries_features(dfs_sticker)
        return df_odds
