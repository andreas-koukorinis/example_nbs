import logging
from datetime import timedelta
from pprint import pprint

import pandas as pd
import pytz

from sgmtradingcore.analytics.abc.odds import check_spread, check_overround, \
    get_tick_bp, get_tick_lp
from sgmtradingcore.analytics.abc.abc import get_connection
from sgmtradingcore.strategies.abc_db.model import MarketTick, NaturalHandicapTS, Sticker


def nat_ou_hc(conn, event_id, market, after_t=None):
    """
    For OU markets, where the opposite selection has the same market selection (e.g. U-1_0 == O-1_0)
    :return [{'timestamp': datetime, 'sel': float}]

    Parameters
    ----------
    conn: (MongoPersister) output of get_connection
    event_id: (str) id of the event to get AH for.
    after_t: (datetime) to get natural handicap ticks only after this time
    """
    query = {
        NaturalHandicapTS.sticker_s1.db_field + '.' + Sticker.event_id.db_field: event_id,
        NaturalHandicapTS.sticker_s1.db_field + '.' + Sticker.market.db_field: market,
    }

    if after_t is not None:
        query[NaturalHandicapTS.timestamp.db_field] = {'$gte': after_t}

    proj = {
        NaturalHandicapTS.timestamp.db_field: 1,
        NaturalHandicapTS.sticker_s1.db_field + '.' + Sticker.param1.db_field: 1,
    }
    coll = conn.sport + '_natural_handicap'
    res = conn[coll].find(query, proj)

    values = [{'timestamp': f[NaturalHandicapTS.timestamp.db_field],
               'sel': f[NaturalHandicapTS.sticker_s1.db_field][Sticker.param1.db_field]}
              for f in res]
    return values


def nat_ou_hc_odds(conn, event_id, market, after_t=None):
    """
    For OU markets, where the opposite selection has the same market selection (e.g. U-1_0 == O-1_0)
    :return [{'timestamp': datetime, 's1': str, 's2': str, 't1': dict, 't2': dict}]

    Parameters
    ----------
    conn: (MongoPersister) output of get_connection
    event_id: (str) id of the event to get AH for.
    after_t: (datetime) to get natural handicap ticks only after this time
    """
    query = {
        NaturalHandicapTS.sticker_s1.db_field + '.' + Sticker.event_id.db_field: event_id,
        NaturalHandicapTS.sticker_s1.db_field + '.' + Sticker.market.db_field: market,
    }

    if after_t is not None:
        query[NaturalHandicapTS.timestamp.db_field] = {'$gte': after_t}

    proj = {
        NaturalHandicapTS.timestamp.db_field: 1,
        NaturalHandicapTS.sticker_s1.db_field + '.' + Sticker.str_.db_field: 1,
        NaturalHandicapTS.sticker_s2.db_field + '.' + Sticker.str_.db_field: 1,
        NaturalHandicapTS.tick_s1.db_field: 1,
        NaturalHandicapTS.tick_s2.db_field: 1,
    }
    coll = conn.sport + '_natural_handicap'
    res = conn[coll].find(query, proj)

    values = [{'timestamp': f[NaturalHandicapTS.timestamp.db_field],
               's1': f[NaturalHandicapTS.sticker_s1.db_field][Sticker.str_.db_field],
               't1': MarketTick.odds_tick_from_dict(f[NaturalHandicapTS.tick_s1.db_field]),
               's2': f[NaturalHandicapTS.sticker_s2.db_field][Sticker.str_.db_field],
               't2': MarketTick.odds_tick_from_dict(f[NaturalHandicapTS.tick_s2.db_field])}
              for f in res]
    return values


def nat_ah_hc(conn, event_id, market, after_t=None):
    """
    For AH markets, where the opposite selection has the opposite market selection (e.g. A-1_0 == B-n1_0)
    return [{'timestamp': datetime, 'sel1': sel1, 'ah1': float), 'sel2': int, 'ah2': float))]

    Parameters
    ----------
    conn: (MongoPersister) output of get_connection
    event_id: (str) id of the event to get AH for.
    after_t: (datetime) to get natural handicap ticks only after this time

    """
    query = {
        NaturalHandicapTS.sticker_s1.db_field + '.' + Sticker.event_id.db_field: event_id,
        NaturalHandicapTS.sticker_s1.db_field + '.' + Sticker.market.db_field: market,
    }

    if after_t is not None:
        query[NaturalHandicapTS.timestamp.db_field] = {'$gte': after_t}

    proj = {
        NaturalHandicapTS.timestamp.db_field: 1,
        NaturalHandicapTS.sticker_s1.db_field + '.' + Sticker.param0.db_field: 1,
        NaturalHandicapTS.sticker_s2.db_field + '.' + Sticker.param0.db_field: 1,
        NaturalHandicapTS.sticker_s1.db_field + '.' + Sticker.param1.db_field: 1,
        NaturalHandicapTS.sticker_s2.db_field + '.' + Sticker.param1.db_field: 1,
    }
    coll = conn.sport + '_natural_handicap'
    res = conn[coll].find(query, proj)

    values = [{'timestamp': f[NaturalHandicapTS.timestamp.db_field],
               'sel1': f[NaturalHandicapTS.sticker_s1.db_field][Sticker.param0.db_field],
               'ah1': f[NaturalHandicapTS.sticker_s1.db_field][Sticker.param1.db_field],
               'sel2': f[NaturalHandicapTS.sticker_s2.db_field][Sticker.param0.db_field],
               'ah2':  f[NaturalHandicapTS.sticker_s2.db_field][Sticker.param1.db_field]}
              for f in res]
    return values


def get_natural_hc_odds(event_id, market_abr, timestamp, abc_db,
                        return_series=True, spread_max=None,
                        overround_max=0.05, delta_stale=timedelta(seconds=30)):
    """
    get handicap, sticker (wrt team/player 1) and back prices from the
    sport-relevant natural handicap abc collection at a snapshot in time

    :param event_id: (str) e.g. 'GSM123456'
    :param market_abr: (str) e.g. 'FTAHG'.
    :param timestamp: (datetime) time at which odds are required.
    :param abc_db: (MongoPersister) connection to abc data base.
    :param return_series: (bool) whether to return pandas Series instead
        of dict.
    :param spread_max: (float) maximum spread tolerated. None
        warning: this will return None if the lay price is missing (e.g. PN)
    :param overround_max: (float) ovr_pct param in check_overround
    :param delta_stale: (timedelta) time after which we discard odds
    :return: (dict or pd.Series) natural handicap from market and market
        information. None if natural handicap not found
    """
    stkr_s1_fld = NaturalHandicapTS.sticker_s1.db_field
    stkr_s2_fld = NaturalHandicapTS.sticker_s2.db_field
    tck_s1_fld = NaturalHandicapTS.tick_s1.db_field
    tck_s2_fld = NaturalHandicapTS.tick_s2.db_field
    t_fld = NaturalHandicapTS.timestamp.db_field

    evt_fld = Sticker.event_id.db_field
    mkt_fld = Sticker.market.db_field

    query = {stkr_s1_fld + '.' + evt_fld: event_id,
             stkr_s1_fld + '.' + mkt_fld: market_abr,
             t_fld: {'$lte': timestamp,
                     '$gte': timestamp - delta_stale}}

    market_projection = {stkr_s1_fld + '.' + Sticker.str_.db_field: 1,
                         stkr_s2_fld + '.' + Sticker.str_.db_field: 1,
                         stkr_s1_fld + '.' + Sticker.param1.db_field: 1,
                         stkr_s2_fld + '.' + Sticker.param1.db_field: 1,
                         tck_s1_fld + '.' + MarketTick.back_price.db_field: 1,
                         tck_s2_fld + '.' + MarketTick.back_price.db_field: 1,
                         tck_s1_fld + '.' + MarketTick.lay_price.db_field: 1,
                         tck_s2_fld + '.' + MarketTick.lay_price.db_field: 1,
                         tck_s1_fld + '.' + MarketTick.timestamp.db_field: 1,
                         tck_s2_fld + '.' + MarketTick.timestamp.db_field: 1,
                         t_fld: 1,
                         '_id': 0}

    coll = abc_db.sport + '_natural_handicap'
    res = list(abc_db[coll].find(
        query, market_projection).sort([(t_fld, -1)]).limit(1))
    if len(res) == 0:
        logging.warning('Could not find natural handicap in abc')
        return None
    res = res[0]

    # check over-round of the two selections
    if overround_max is not None:
        ticks = [res[tck_s1_fld], res[tck_s2_fld]]
        if not check_overround(ticks, arb_pct=0.02, ovr_pct=overround_max,
                               is_back=True):
            logging.debug('Over-round (or arb opportunity) too big')
            return None

    # check and format the ticks
    for tck_side_fld in [tck_s1_fld, tck_s2_fld]:
        tick_side = res[tck_side_fld]
        # check that the ticks are not stale
        delta_tick = timestamp - tick_side[MarketTick.timestamp.db_field]
        if delta_tick > delta_stale:
            logging.debug('Tick for {} is stale by {}'.format(
                tck_s1_fld, delta_tick))
            return None

        # check that the spread is not too big
        if spread_max is not None and not check_spread(tick_side, spread_max):
            logging.debug('Spread for {} was too big'.format(tck_side_fld))
            return None

        # convert odds into decimals from their storage int form
        tick_side[MarketTick.back_price.db_field] = get_tick_bp(tick_side)
        tick_side[MarketTick.lay_price.db_field] = get_tick_lp(tick_side)

    # rename from db name so that they do not depend on db model
    out = {
        'stkr_s1': {
            'p1': res[stkr_s1_fld][Sticker.param1.db_field],
            'str': res[stkr_s1_fld][Sticker.str_.db_field]
        },
        'stkr_s2': {
            'p1': res[stkr_s2_fld][Sticker.param1.db_field],
            'str': res[stkr_s2_fld][Sticker.str_.db_field]
        },
        't': res[NaturalHandicapTS.timestamp.db_field],
        'tck_s1': {
            'bp': res[tck_s1_fld][MarketTick.back_price.db_field],
            'lp': res[tck_s1_fld][MarketTick.lay_price.db_field],
            't': res[tck_s1_fld][MarketTick.timestamp.db_field]
        },
        'tck_s2': {
            'bp': res[tck_s2_fld][MarketTick.back_price.db_field],
            'lp': res[tck_s2_fld][MarketTick.lay_price.db_field],
            't': res[tck_s2_fld][MarketTick.timestamp.db_field]
        }
    }

    if return_series:
        out = pd.Series(_flatten(out))
        out.name = event_id

    return out


def _flatten(d, parent_key=''):
    """
    flatten nested dicts
    """
    items = []
    for k, v in d.items():
        try:
            items.extend(_flatten(v, '%s%s_' % (parent_key, k)).items())
        except AttributeError:
            items.append(('%s%s' % (parent_key, k), v))
    return dict(items)


def main_example_get_nat_hc_odds():
    """
    example for get_fball_natural_hc_odds()
    """
    from datetime import datetime

    timestamp = datetime(2018, 5, 5, 15, 30, tzinfo=pytz.UTC)
    abc_db = get_connection('football')
    o = get_natural_hc_odds(
        event_id='GSM2463159',
        market_abr='FTOUC',
        timestamp=timestamp,
        abc_db=abc_db,
        delta_stale=timedelta(seconds=250),
    return_series=True)
    pprint(o)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    conn = get_connection('football')

    main_example_get_nat_hc_odds()
    # res_ts = nat_ah_hc(conn, 'GSM2465138', 'FTAHG')
    # print pd.DataFrame(res_ts)

    # # print get_sticker(conn, 'T-EENP1923578-FT12-A.BF')
    # cache = FixtureCache()
    # kickoff = cache.get_kickoff('ENP2191253')
    #
    # print nat_ou_hc(conn, 'GSM1487207', 'FTOUG')
    # print nat_ou_hc_odds(conn, 'ENP2180538', 'TGOU')
    #
    # print nat_ah_hc(conn, 'ENP2180538', 'HG')
