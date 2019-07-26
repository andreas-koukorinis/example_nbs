import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sgmtradingcore.analytics.features.feature import TimeSeriesFeature, EventFeature
from sgmtradingcore.analytics.features.storage import (InMemoryFeatureStorage, MongoCompactTimeseriesStorage,
                                                       MongoEventFeatureStorage)
from sgmtradingcore.providers.odds_providers import merge_markets, OddsTick
from sgmtradingcore.util.misc import chunks
from stratagemdataprocessing.bookmakers.common.odds.cache import s3_sticker_list_from_fixture_id
from stratagemdataprocessing.enums.odds import Bookmakers, Sports
from stratagemdataprocessing.enums.markets import TennisMarkets, BasketballMarkets, Markets
from stratagemdataprocessing.bookmakers.common.odds.fast_odds import OddsTickOpt
from stratagemdataprocessing.parsing.common.stickers import BOOKMAKER_ABBR
from stratagemdataprocessing.parsing.common.stickers import sticker_parts_from_sticker


def compute_book_pressure(tick, input_weights):
    """
    Return a value [-1, 1] which is Vb-Vl/Vb+Vl
    If Vb == Vl == 0 returns 0.0 .
    If Vl==0 return +1.0
    If Vb==0 return -1.0

    Ignore negative volumes

    :param tick: type OddsTick or OddsTickOpt,
    :param input_weights: type [float], volumes are weights by this amount. If input_weight is shorter than the depth
                          of the market, will be p[added with zeroes; is shorter, will be truncated.
    :return: float
    """
    bv = [o.v for o in tick.back if o.v > 0]
    weights = copy.copy(input_weights)
    if len(bv) > len(weights):
        weights = np.concatenate((weights, np.zeros(len(bv) - len(weights))))
    bv_ = np.dot(bv, weights[:len(bv)])

    lv = [o.v for o in tick.lay if o.v > 0]
    weights = copy.copy(input_weights)
    if len(lv) > len(weights):
        weights = np.concatenate((weights, np.zeros(len(lv) - len(weights))))
    lv_ = np.dot(lv, weights[:len(lv)])

    if 0 == len(bv) == len(lv):
        return .0
    return np.true_divide(bv_ - lv_, bv_ + lv_)


def get_odds_df(sticker, bookmakers, odds_cache, remove_negatives=True):
    """
    Return a timeseries of OddsTick which is the merge of all bookmakers market.
    :return: DataFrame indexed by timestamp, where every row is a OddsTicks with the merged OddsTicks for all
             the selected bookmakers.
    """
    _sticker = sticker.split('.')[0]
    tick_dfs = list()

    for bm in [BOOKMAKER_ABBR[b] for b in bookmakers]:
        sticker = "{}.{}".format(_sticker, bm)
        ticks = odds_cache.get(sticker) or []
        index = [t.timestamp for t in ticks]
        d = DataFrame(data={sticker: ticks}, index=index)
        d.index.name = 'timestamp'
        tick_dfs.append(d)

    df = tick_dfs[0].copy()
    for d in tick_dfs[1:]:
        df = pd.merge(df, d, left_index=True, right_index=True, how='outer', sort=True).fillna(method='ffill')

    # remove duplicates if any
    df = df[~df.index.duplicated(keep='last')]

    combined_odds = []
    for index, row in df.iterrows():
        market_snapshot = {s: o for s, o in row.to_dict().iteritems() if
                           isinstance(o, OddsTickOpt) or isinstance(o, OddsTick)}
        mm = merge_markets(index, market_snapshot, _sticker, bookmakers, remove_negatives=remove_negatives)
        combined_odds.append(mm)

    combined_df = DataFrame(index=df.index, data={_sticker: combined_odds})
    combined_df.index.name = 'timestamp'

    return combined_df


class AvailableStickersSingleBookmaker(EventFeature):
    REQUIRED_PARAMETERS = {'bookmaker'}

    def __init__(self, runner, params, **kwargs):
        """
        Lists stickers available on S3 for an event for a single bookmaker. Cached in Mongo.
        :param runner:
        :param params: bookmaker
        :param kwargs:
        """
        self._bookmaker = params['bookmaker']
        self._odds_cache = runner.shared_objects()['odds_cache']
        self._mongo_connection = runner.shared_objects().get('mongo_connection', None)
        self._mysql_connection = runner.shared_objects().get('mysql_client', None)

        storage = InMemoryFeatureStorage(self, parent_cache=MongoEventFeatureStorage(self,
                                                                                     mongo_connection=self._mongo_connection))
        super(AvailableStickersSingleBookmaker, self).__init__(runner, params, storage=storage, **kwargs)

    @staticmethod
    def _key_to_sticker(key):
        name = key.key
        return name.split('/')[-1][5:-7]

    def _compute_by_event_id(self, event_id, **kwargs):
        stickers = s3_sticker_list_from_fixture_id(event_id, [self._bookmaker], self._odds_cache,
                                                   self._mysql_connection)

        return {'stickers_{}'.format(self._bookmaker): stickers}


class AvailableStickers(EventFeature):

    REQUIRED_PARAMETERS = {'bookmakers'}

    def __init__(self, runner, params, **kwargs):
        """
        Returns all stickers available on S3 for an event, not cached - Uses AvailableStickersSingleBookmaker
        to cache in mongo
        :param runner:
        :param params: [list[Bookmaker]] i.e. Bookmaker.PINNACLE_SPORTS...
        :param kwargs:
        """

        super(AvailableStickers, self).__init__(runner, params, storage=None, **kwargs)
        self._bookmakers = self._params['bookmakers']
        pass

    def _compute_by_event_id(self, fixture_id, repopulate=False):
        """
        :param fixture_id:
        :param repopulate: Repopulote the child features
        :return:
        """
        feature_requests = [FeatureRequest('AvailableStickersSingleBookmaker',
                                                         {'bookmaker': b}, {})
                            for b in self._bookmakers]

        result = self._runner.get_event_features(feature_requests, [fixture_id], repopulate=repopulate)
        return {
            'stickers': [s for sticker_list in result[fixture_id]['AvailableStickersSingleBookmaker'].itervalues() for s
                         in sticker_list]}


class MarketFeature(TimeSeriesFeature):
    REQUIRED_PARAMETERS = {'bookmakers'}

    def __init__(self, runner, params, storage=None, **kwargs):
        """
        Base class for markets timeseries. Use it if you want to return a timeserie of processed market data coming from
        the odds_cache.
        Prefetch stickers data from odds_cache.

        :param runner:
        :param params: type dict:
            'bookmakers': type [list[Bookmakers]] Bookmakers.BETFAIR etc
        'markets': type [list[int]] Market Enums
        :param kwargs:
            'precache' : [optional] pre fetch odds cache; useful if you need to recompute for a huge number of stickers
        """
        super(MarketFeature, self).__init__(runner, params, storage=storage, **kwargs)
        self._bookmakers = params['bookmakers']
        self._bookmakers_abbr = [BOOKMAKER_ABBR[bm] for bm in params['bookmakers']]

        self._odds_cache = runner.shared_objects()['odds_cache']
        self._precache = kwargs.get('precache', False)

    @classmethod
    def get_feature_id(cls, feature_class_name, params, runner, feature_conn=None, force_new=False):
        """
        Makes feature_id independent from the bookmakers order
        """
        par = copy.deepcopy(params)
        par['bookmakers'] = sorted(par['bookmakers'])
        return super(MarketFeature, cls).get_feature_id(feature_class_name, par, runner,
                                                        feature_conn=feature_conn, force_new=force_new)

    def initialize_stickers(self, stickers):
        if self._precache:
            stickers = list({s.split('.')[0] for s in stickers})
            logging.info("Pre-caching {} stickerfiles".format(len(stickers) * len(self._bookmakers_abbr)))
            for chunk in chunks(stickers, 200):
                chunk_with_bm = ['{}.{}'.format(s, bm) for s in chunk for bm in self._bookmakers_abbr]
                self._odds_cache.fetch_list(chunk_with_bm)

    def initialize_events(self, event_ids):
        if self._precache:
            for event_id in event_ids:
                stickers = self._get_stickers_for_match(event_id)
                self.initialize_stickers(stickers)

    def _get_stickers_for_match(self, event_id):
        request = FeatureRequest('AvailableStickers', {'bookmakers': self._bookmakers}, {})
        stickers = self._runner.get_event_features([request], [event_id])
        return stickers[event_id]['AvailableStickers']['stickers']

    def _compute_by_event_id(self, fixture_id, repopulate=False):
        stickers = self._get_stickers_for_match(fixture_id)
        return self.compute_for_stickers(stickers, repopulate=repopulate)

    def get_df_from_event_id(self, event_id, repopulate=False, recompute_missing=True):
        """
        Get a dataframe with one or more columns, where the column name typically is NOT the event_id e.g. could be
        one column per sticker or call the column as the feature name
        :param event_id: type str, like 'ENP222222' or 'GSM33333'
        :param repopulate: repopulate the feature for this event_id but not its dependencies
        :return:
        """
        stickers = self._get_stickers_for_match(event_id)
        if repopulate:
            for sticker in stickers:
                self._storage.delete_feature_by_sticker(self.feature_id, sticker)
            df = None
        else:
            dfs = []
            for sticker in stickers:
                df = self._storage.load_feature_by_sticker(self.feature_id, sticker)
                if df is not None:
                    df.columns = [sticker]
                    dfs.append(df)
            if dfs:
                df = pd.concat(dfs)
            else:
                df = None

        if df is None and recompute_missing:
            df = self._compute_by_event_id(event_id, repopulate=repopulate)
            if df is not None:
                self._storage.store_feature_by_event_id(self.feature_id, event_id, df)
            else:
                df = self.get_empty_feature()
        return df


class BookPressure(MarketFeature):
    """
    Return a value [-1, 1] which is Vb-Vl/Vb+Vl averaged across many bookmakers.
    If Vb == Vl == 0 return NaN.
    If Vl==0 return +1.0
    If Vb==0 return -1.0
    Aims to give an idea of which side of the LOB is more 'heavy'
    so if all the weight is on one side (lots of people trying to back)
    it could be an indicator that price will decrease.

    It does no use synthetic prices, e.g. do not sum volume for back FTOUG-O-2_5 with lay FTOUG-U-2_5.
    BookPressure ignores prices and negative volumes

    """
    REQUIRED_PARAMETERS = {'weights', 'decay'}
    def __init__(self, runner, params, **kwargs):
        """

        :param runner:
        :param params: type dict:
            'bookmakers': type [str] e.g. ['BF', 'PN', 'MB']. The order is relevant, otherwise change feature_id
            'decay': type float<=1, each weight will be decay**level
            'weights': (optional) type [Float], weighted volumes are taken.
                       If the array is too short, will be filled with zeroes. If 'weights is present, 'decay' is ignored
        :param kwargs:
            'precache' : [optional] pre fetch odds cache; useful if you need to recompute for a huge number of stickers
        """

        storage = MongoCompactTimeseriesStorage(self)
        super(BookPressure, self).__init__(runner, params, storage=storage, **kwargs)

        if params.get('weights', None) is not None:
            self._weights = params['weights']
        else:
            decay = params['decay']
            self._weights = [decay ** i for i in range(1, 11)]

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'decay': 0.91,
                'weights': None}

    def _compute_by_sticker(self, sticker):
        """
        Compute the feature to be stored in cache.
        Override this method.

        You have access to the FeatureRunner from here
        :return a DataFrame, where the column is the name of the feature. Can contain multiple columns.
        """

        _sticker = sticker.split('.')[0]
        ticks = get_odds_df(_sticker, self._bookmakers, self._odds_cache, remove_negatives=True)[_sticker]

        data = [compute_book_pressure(tick, self._weights) for tick in ticks]
        index = [tick.timestamp for tick in ticks]

        out = DataFrame(
            data={self.__class__.__name__: data},
            index=[index])
        out.index.name = 'timestamp'

        return out

    def compute_for_stickers(self, stickers, repopulate=False):
        missing_stickers = self._storage.missing_stickers(self.feature_id, stickers)
        logging.info("Missing {} stickers".format(len(missing_stickers)))
        for sticker in tqdm(missing_stickers):
            df = self._compute_by_sticker(sticker)
            if df is not None:
                self._storage.store_feature_by_sticker(self.feature_id, sticker, df)
            else:
                df = self.get_empty_feature()
        return


def compute_micro_price(tick):
    """
    SUM_levels (Vl*Pb + Vb*Pl)/ SUM_levels Vb+Vl

    Ignores negative volumes.

    Problem: very high odds screw the computation, e.g.:
        back: 18@2.86,  8@2.60, 21@1.23, lay: 18@4.50,  3@990.00 -> 118.91391253
    :param tick: type OddsTick or OddsTickOpt,
    :return: float
    """
    max_i = min(len(tick.back), len(tick.lay))
    b_ = 0
    l_ = 0
    v_ = 0
    for i in range(max_i):
        b_ += tick.lay[i].v * tick.back[i].o
        l_ += tick.back[i].v * tick.lay[i].o
        v_ += tick.back[i].v + tick.lay[i].v
    if v_ == 0:
        return np.nan
    else:
        return (b_ + l_) / float(v_)


def compute_micro_price_flipped(tick):
    """
    SUM_levels (Vb*Pb + Vl*Pl)/ SUM_levels Vb+Vl

    Ignores negative volumes

    Problem: very high odds screw the computation, e.g.:
        back: 18@2.86,  8@2.60, 21@1.23, lay: 18@4.50,  3@990.00  -> 43.5841119976
    :param tick: type OddsTick or OddsTickOpt,
    :return: float
    """
    b_ = sum([t.v * t.o for t in tick.back if t.v > 0])
    l_ = sum([t.v * t.o for t in tick.lay if t.v > 0])
    vols = sum([t.v for t in tick.back + tick.lay if t.v > 0])
    r = np.true_divide(b_ + l_, vols)
    return r


class MicroPrice(MarketFeature):
    """
    Return the micro_price averaged across many bookmakers
    """
    REQUIRED_PARAMETERS = {'flipped'}

    def __init__(self, runner, params, **kwargs):
        """

        :param runner:
        :param params: type dict:
            'bookmakers': type [str] e.g. ['BF', 'PN', 'MB']. The order is relevant, otherwise change feature_id
            'flipped': use the flipped one instead
        :param kwargs:
            'precache': [optional] pre fetch odds cache; useful if you need to recompute for a huge number of stickers
        """
        storage = MongoCompactTimeseriesStorage(self)
        super(MicroPrice, self).__init__(runner, params, storage=storage, **kwargs)
        self._flipped = params['flipped']

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'flipped': False}

    def _compute_by_sticker(self, sticker):
        """
        Compute the feature to be stored in cache.
        Override this method.

        You have access to the FeatureRunner from here
        :return a DataFrame, where the column is the name of the feature. Can contain multiple columns.
        """

        _sticker = sticker.split('.')[0]
        ticks = get_odds_df(_sticker, self._bookmakers, self._odds_cache, remove_negatives=True)[_sticker]

        if self._flipped:
            data = [compute_micro_price_flipped(tick) for tick in ticks]
        else:
            data = [compute_micro_price(tick) for tick in ticks]
        index = [tick.timestamp for tick in ticks]

        colname = 'MicroPrice' if not self._flipped else 'MicroPrice_flipped'
        out = DataFrame(data={colname: data}, index=[index])
        out.index.name = 'timestamp'
        out.sort_index()

        return out


if __name__ == "__main__":
    import logging
    # from stratagemdataprocessing.bookmakers.dq_check.odds import FOOTBALL_BOOKMAKERS
    from sgmtradingcore.analytics.features.runner import FeatureRunner
    from sgmtradingcore.analytics.features.request import FeatureRequest

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - ' '%(levelname)s - %(message)s')

    # These are the shared objects than can be used by all the Features.

    #
    #
    # bookmakers = FOOTBALL_BOOKMAKERS
    #
    # stickers = ['S-EGSM2465098-FTOUG-O-2_25', 'BB-EENP2353793-FHPS-1-n9_0', 'T-EENP921854-FT12-B']
    # runner = FeatureRunner()
    #
    # requests = [
    #     FeatureRequest(BookPressure.__name__,
    #                    {'bookmakers': FOOTBALL_BOOKMAKERS},
    #                    # 'pressure_',
    #                    {'precache': True}, ),
    #     FeatureRequest(MicroPrice.__name__,
    #                    {'bookmakers': FOOTBALL_BOOKMAKERS},
    #                    {'precache': True}),
    #     FeatureRequest(MicroPrice.__name__,
    #                    {'bookmakers': FOOTBALL_BOOKMAKERS, 'flipped': True},
    #                    {'precache': True}),
    # ]
    # dfs = runner.get_dataframes_by_stickers(requests, stickers)
    # # dfs = runner.get_dataframes_by_stickers_multithread(requests, stickers, n_threads=6)
    #
    # print 'returned {} dfs'.format(len(dfs))
    # print dfs
    requests = [
        FeatureRequest('AvailableStickers', {}, {}),
        FeatureRequest('AvailableStickers',
                       {'bookmakers': [Bookmakers.PINNACLE_SPORTS, Bookmakers.MATCHBOOK, Bookmakers.BETFAIR]},
                       {},
                       prefix='a_'),
        FeatureRequest('AvailableStickers',
                       {'bookmakers': [Bookmakers.PINNACLE_SPORTS, Bookmakers.MATCHBOOK, Bookmakers.BETFAIR]},
                       {},
                       prefix='b_')
    ]

    events = {u'ENP2613971', u'ENP2613972', u'ENP2613973', u'ENP2613974', u'ENP2613975', u'ENP2613976', u'ENP2613977',
              u'ENP2613979', u'ENP2613980', u'ENP2613981', u'ENP2613982', u'ENP2613983', u'ENP2613985', u'ENP2613986',
              u'ENP2613987', u'ENP2613988', u'ENP2613989'}

    runner = FeatureRunner()

    stickers = runner.get_event_features(requests, events)
    # dfs = runner.get_dataframes_by_stickers_multithread(requests, stickers, n_threads=6)
    print stickers
