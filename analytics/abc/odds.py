import copy
import logging
from datetime import timedelta
import numpy as np
from stratagemdataprocessing.events.fixture_cache import FixtureCache
from stratagemdataprocessing.enums.odds import SelectionStatus
from sgmtradingcore.analytics.abc.abc import get_connection
from sgmtradingcore.strategies.abc_db.model import ABCMarketData, MarketTick, \
    Sticker, SCALE_FACTOR

ACTIVE_STATUS = SelectionStatus.to_str_symbol(SelectionStatus.ACTIVE)
SUSPENDED_STATUS = SelectionStatus.to_str_symbol(SelectionStatus.SUSPENDED)

# Each market data segment includes a boolean field `deadball` which is `True` if
# all ticks within the segment correspond to deadball time and `False` if at
# least one tick in the segment corresponds to in-play time.


def get_sticker_odds(conn, sticker):
    """
    :return: [{}] in MarketTick dictformat
    """
    query = {ABCMarketData.sticker.db_field+'.'+Sticker.str_.db_field: sticker}
    proj = {ABCMarketData.ticks.db_field: 1}
    res = conn[conn.sport+'_market_data'].find(query, proj)

    ticks = []
    for r in res:
        ticks.extend(r[ABCMarketData.ticks.db_field])

    # check timestamps are incremental
    # for i, t in enumerate(ticks[1:]):
    #     assert ticks[i-1][MarketTick.timestamp.db_field] <= ticks[i-1][MarketTick.timestamp.db_field]

    return ticks


def _fetch_last_tick_before(conn, sticker, dt):
    query = {
        ABCMarketData.sticker.db_field + '.' + Sticker.str_.db_field: sticker,
        ABCMarketData.timestamp.db_field: {"$lte": dt}
    }
    proj = {ABCMarketData.ticks.db_field: 1}
    return conn[conn.sport + '_market_data'].find(query, proj).sort('_id', -1).limit(1)


def get_last_tick_before(conn, sticker, dt):
    """
    Get the last tick <= dt
    :return {} in MarketTick dict format or None
    """
    it = _fetch_last_tick_before(conn, sticker, dt)
    values = [f for f in it]
    if len(values) == 0:
        return None

    ticks = values[0][ABCMarketData.ticks.db_field]
    ret = None
    for tick in ticks:
        if tick[MarketTick.timestamp.db_field] < dt:
            ret = tick
        else:
            break

    return ret


def get_first_in_play_tick_from_market(conn, sticker):
    """
    :return: {} or None
    """
    col_mkt = conn[conn.sport + '_market_data']
    query_dbl = {ABCMarketData.sticker.db_field + '.' + Sticker.str_.db_field: sticker,
                 ABCMarketData.deadball.db_field: False}
    project_dbl = {ABCMarketData.ticks.db_field: 1}
    mkt = [i for i in col_mkt.find(query_dbl, project_dbl).limit(1)]
    if not len(mkt):
        return None
    mkt = mkt[0]
    for tk in mkt[ABCMarketData.ticks.db_field]:
        if not tk[MarketTick.deadball.db_field]:
            return tk
    else:  # goes here if does not break
        raise Exception('All the ticks are dbl but global dbl is False')


def expire_abc_market_tick(tick, timestamp, timeout_ms=None, expire_in_play_only=False):
    """
    Invalidate every tick after timeout_ms; adds ticks with SUSPENDED status


    :param tick: [{}] as in abc
    :param timestamp: the current timestamp
    :param timeout_ms: int. If None do not apply
    :param expire_in_play_only: if True only expire the tick if it has positive delay.
    :return : {}
    """

    if (timestamp -
        tick[MarketTick.timestamp.db_field]).total_seconds() * 1000 > timeout_ms and \
            tick[MarketTick.status.db_field] == ACTIVE_STATUS:

        if expire_in_play_only and tick[MarketTick.deadball.db_field]:
            return None
        else:
            new_timestamp = tick[MarketTick.timestamp.db_field] + timedelta(milliseconds=timeout_ms)

        new_tick = copy.deepcopy(tick)
        new_tick[MarketTick.status.db_field] = SUSPENDED_STATUS
        new_tick[MarketTick.timestamp.db_field] = new_timestamp
        return new_tick

    else:
        return None


def sanitize_sticker_odds_abc(ticks, timeout_ms=None, expire_in_play_only=False):
    """
    Invalidate the MarketTick after timeout_ms; adds ticks with SUSPENDED status

    :param ticks: ({} as returned by ABC)
    :param timeout_ms: int. If None do not apply
    :param expire_in_play_only: if True only expire the ticks with positive delay.
            If no ticks with positive delay exists, expire all of them
    :return ([{}]):
    """
    if timeout_ms is None or len(ticks) == 0:
        return ticks

    ret_ticks = [ticks[0]]
    for tick in ticks[1:]:
        new_tick = expire_abc_market_tick(ret_ticks[-1], tick[MarketTick.timestamp.db_field],
                                          timeout_ms=timeout_ms, expire_in_play_only=expire_in_play_only)
        if new_tick is not None:
            ret_ticks.append(new_tick)
        ret_ticks.append(tick)

    # expire last tick
    new_tick = expire_abc_market_tick(ticks[-1],
                                      ticks[-1][MarketTick.timestamp.db_field]+timedelta(milliseconds=timeout_ms+1),
                                      timeout_ms=timeout_ms, expire_in_play_only=expire_in_play_only)
    if new_tick is not None:
        ret_ticks.append(new_tick)

    return ret_ticks


def get_sanitized_sticker_odds(conn, sticker, expire_in_play_only=False):
    ticks = get_sticker_odds(conn, sticker)
    sanitized = sanitize_sticker_odds_abc(ticks, timeout_ms=30000, expire_in_play_only=expire_in_play_only)
    return sanitized


def check_spread(tick, spread_max):
    """Check that the market spread is not too big and that prices are decent

    Parameters
    ----------
    tick: dict, tick from abc market data
    spread_max: float, difference between bp1 and lp1 in probability terms

    Returns
    -------
    True if the spread is strictly less than spread_max, False otherwise.
    """
    bp = tick[MarketTick.back_price.db_field]
    lp = tick[MarketTick.lay_price.db_field]

    spread = abs(SCALE_FACTOR / bp - SCALE_FACTOR / lp)
    logging.debug('Spread is {} - {} = {}'.format(lp, bp, spread))
    check_spread_ = (1.01 < bp / SCALE_FACTOR < 100 and
                    1.01 < lp / SCALE_FACTOR < 100. and
                    spread < spread_max)
    return check_spread_


def check_overround(ticks, arb_pct=0.01, ovr_pct=0.05, is_back=True):
    """Check over-round of the market ticks

    Parameters
    ----------
    ticks: list, ticks of all the selections for the same market
    arb_pct: float, arbitrage % threshold for detecting wrong prices
    ovr_pct: float, over-round % threshold for ignoring too big over-round
    is_back: boolean,  whether we are looking at the back or lay selection

    Returns
    -------
    whether the prices are sensible in term of probability of the event
    happening are close to 100
    """
    p_name = MarketTick.back_price.db_field if is_back else (
        MarketTick.lay_price.db_field)
    proba = np.sum([SCALE_FACTOR / float(t[p_name]) for t in ticks])
    if not is_back:
        proba = 2. - proba
    proba_inf = 1. - arb_pct
    proba_sup = 1. + ovr_pct
    return proba_inf <= proba <= proba_sup


def get_tick_lbp(tick):
    """
    return a dictionary containing lp and bp prices in keys "lp" and "bp"
    """
    if (tick is None or tick[MarketTick.back_price.db_field] < 1 * SCALE_FACTOR
       or tick[MarketTick.lay_price.db_field] < 1 * SCALE_FACTOR):
        return None
    return {'bp': tick[MarketTick.back_price.db_field] / SCALE_FACTOR,
            'lp': tick[MarketTick.lay_price.db_field] / SCALE_FACTOR}


def get_tick_bp(tick):
    """
    Parameters
    ----------
    tick: (dict) tick from abc market data

    Returns
    -------
    - back price rescaled
    """
    if tick is None or tick[MarketTick.back_price.db_field] < 1 * SCALE_FACTOR:
        return None
    return tick[MarketTick.back_price.db_field] / SCALE_FACTOR


def get_tick_lp(tick):
    """
    Parameters
    ----------
    tick: (dict) tick from abc market data

    Returns
    -------
    - lay price rescaled
    """
    if tick is None or tick[MarketTick.lay_price.db_field] < 1 * SCALE_FACTOR:
        return None
    return tick[MarketTick.lay_price.db_field] / SCALE_FACTOR


def get_tick_mid_price(tick):
    bp = get_tick_bp(tick)
    lp = get_tick_lp(tick)
    if bp is None or lp is None:
        return None
    else:
        return (bp + lp) / 2.


if __name__ == '__main__':
    conn = get_connection('tennis')
    # print get_sticker_odds(conn, 'T-EENP1923578-FT12-A.BF')
    cache = FixtureCache()
    kickoff = cache.get_kickoff('ENP2191253')
    # print get_sanitized_sticker_odds(conn, 'T-EENP2191253-FT12-A.BF', kickoff)
    print get_last_tick_before(conn, 'T-EENP2191253-FT12-A.BF', kickoff)

    # print existing_stickers_from_meta(conn, 'ENP1923578')

