import logging
from collections import namedtuple

from sgmtradingcore.providers.odds_providers import OddsTick, OddsStream, get_historical_odds
from stratagemdataprocessing.parsing.common.stickers import parse_sticker, BOOKMAKER_ABBR

_trade = namedtuple('trade', ('size', 'price', 'delay', 'total'))


def market_to_trades(market):
    # we need at least 2 ticks to get a baseline total matched volume
    if len(market) < 2:
        return []

    trades = []

    last_total = market[0].total
    for market_tick in market[1:]:
        if market_tick.total > last_total:
            trade_size = market_tick.total - last_total
            traze_price = market_tick.lpm

            trades.append(_trade(trade_size, traze_price, market_tick.delay, market_tick.total))

            last_total = market_tick.total

    return trades


def filter_market(market, start_dt, end_dt, no_delay=False, include_start_tick=False):
    start_tick = None  # the tick before the time window to include if include_start_tick
    last_tick = None  # the previous tick in each iteration
    result = []

    for i, market_tick in enumerate(market):

        if no_delay and market.delay > 0:
            continue

        is_start_tick = start_dt <= market_tick.timestamp and start_tick is None
        is_first_iteration = i == 0

        if is_start_tick and not is_first_iteration and include_start_tick:
            start_tick = last_tick
            result.append(start_tick)

        if start_dt <= market_tick.timestamp <= end_dt:
            start_tick = OddsTick(None, None)
            result.append(market_tick)

        last_tick = market_tick

    return result


def vwap(trades):
    """
    Volume weighted average price, ie sumproduct price, size / sum size
    """
    result = 0
    total = 0

    if len(trades) == 0:
        return 0

    for trade in trades:
        result += trade.size * trade.price
        total += trade.size

    result = result / total

    return result


def twabp(odds_ticks):
    """
    Time weighted average back price, note this is not really *weighted* by time
    ie. sum price / count price
    """
    sum_price = 0
    n_price = 0

    for odds_tick in odds_ticks:
        if len(odds_tick.back) > 0:
            sum_price += odds_tick.back[0].o
            n_price += 1

    result = sum_price / n_price

    return result


def exchange_vwap(sticker, bookmaker, start_dt, end_dt, odds_cache=None, return_nan=False):
    try:
        sport, market_scope, market, params, _ = parse_sticker(sticker)
        stream = OddsStream(sport, market_scope[1], market, *params, bookmaker=bookmaker)

        odds = get_historical_odds([stream], odds_cache=odds_cache, start_dt=start_dt, end_dt=end_dt)[stream]
        trades = market_to_trades(odds)
        vwap_ = vwap(trades)
    except (TypeError, ValueError, ZeroDivisionError) as e:
        logging.warn('Could not calculate vwap for sticker %s.%s: %s' % (sticker, BOOKMAKER_ABBR[bookmaker], e))
        if return_nan:
            return float('nan')
        else:
            raise

    return vwap_


def bookmaker_twabp(sticker, bookmaker, start_dt, end_dt, odds_cache=None, return_nan=False):
    try:
        sport, market_scope, market, params, _ = parse_sticker(sticker)
        stream = OddsStream(sport, market_scope[1], market, *params, bookmaker=bookmaker)

        odds = get_historical_odds([stream], odds_cache=odds_cache, start_dt=start_dt, end_dt=end_dt)[stream]
        twabp_ = twabp(odds)
    except (TypeError, ValueError, ZeroDivisionError) as e:
        logging.warn('Could not calculate twabp for sticker %s.%s: %s' % (sticker, BOOKMAKER_ABBR[bookmaker], e))
        if return_nan:
            return float('nan')
        else:
            raise

    return twabp_