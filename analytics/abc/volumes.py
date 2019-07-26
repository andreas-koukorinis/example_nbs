import pytz
from datetime import datetime
import operator

from sgmtradingcore.analytics.abc.abc import get_connection
from sgmtradingcore.analytics.abc.odds import get_sticker_odds, get_last_tick_before, ACTIVE_STATUS
from sgmtradingcore.strategies.abc_db.model import MarketTick, SCALE_FACTOR


def _get_matched_volume_between_oddstick(tick_a, tick_b):
    """
    :param tick_a: OddsTick
    :param tick_b: OddsTick
    """
    va = max(0, tick_a.total) if tick_a is not None else 0.0
    vb = max(0, tick_b.total) if tick_b is not None else 0.0

    return vb - va


def _get_matched_volume_between(tick_a, tick_b):
    va = max(0, tick_a[MarketTick.total.db_field]) if tick_a is not None else 0.0
    vb = max(0, tick_b[MarketTick.total.db_field]) if tick_b is not None else 0.0

    return (vb - va) / SCALE_FACTOR


def get_matched_volume_between(conn, sticker, datetime_from, datetime_to):
    """
    Total volume matched on sticker sticker between timestamps datetime_from and datetime_to
    """

    a = get_last_tick_before(conn, sticker, datetime_from)
    b = get_last_tick_before(conn, sticker, datetime_to)
    return _get_matched_volume_between(a, b)


def _take_vol(price_levels, limit_price, volume, op):
    """

    :param limit_price:
    :param volume:
    :param price_levels: [(bp1, v1), (bp1, v2)] or [(lp1, v1), (lp1, v2)]
    :return:
    """
    vol_taken = 0
    avg_price_taken = 0
    vol_to_take = volume

    for level in price_levels:
        p, v = level[0], level[1]
        if vol_to_take <= 0:
            break
        if p <= 0 or v <= 0:
            continue
        if op(limit_price, p):
            to_take = min(v, vol_to_take)
            new_avg_price = (to_take*p + vol_taken*avg_price_taken) / (to_take + vol_taken)
            avg_price_taken = new_avg_price
            vol_taken += to_take
            vol_to_take -= to_take
    return avg_price_taken, vol_taken


def _take_back_vol(tick, limit_price, volume):
    price_levels = [
        (tick[MarketTick.back_price.db_field], tick[MarketTick.back_volume.db_field]),
        (tick[MarketTick.back_price_2.db_field], tick[MarketTick.back_volume_2.db_field]),
        (tick[MarketTick.back_price_3.db_field], tick[MarketTick.back_volume_3.db_field]),
    ]
    return _take_vol(price_levels, limit_price, volume, operator.le)


def _take_lay_vol(tick, limit_price, volume):
    price_levels = [
        (tick[MarketTick.lay_price.db_field], tick[MarketTick.lay_volume.db_field]),
        (tick[MarketTick.lay_price_2.db_field], tick[MarketTick.lay_volume_2.db_field]),
        (tick[MarketTick.lay_price_3.db_field], tick[MarketTick.lay_volume_3.db_field]),
    ]
    return _take_vol(price_levels, limit_price, volume, operator.ge)


def _game_avail_volume(tick, limit_price, volume, is_back):
    if tick[MarketTick.status.db_field] != ACTIVE_STATUS:
        return .0, .0
    limit_price = limit_price * SCALE_FACTOR
    volume = volume * SCALE_FACTOR
    if is_back:
        price, vol = _take_back_vol(tick, limit_price, volume)
    else:
        price, vol = _take_lay_vol(tick, limit_price, volume)
    return round(price/SCALE_FACTOR, 3), round(vol/SCALE_FACTOR, 3)


def get_volume_at_price(ticks, limit_price, volume, is_back):
    """
    'Which price and volume would I get for V volume, P price on sticker S'
    :return [{'o': float, 'v': float, 't': datetime]: (timestamp, price you would get, volume you would get)
    """
    ret_ticks = list()

    for tick in ticks:
        o, v = _game_avail_volume(tick, limit_price, volume, is_back)
        ret_ticks.append({'o': o, 'v': v, 't': tick[MarketTick.timestamp.db_field]})

    return ret_ticks


def get_price_for_volume_series(conn, sticker, limit_price, volume, is_back):
    """

    'Which price and volume would I get for V volume, P price on sticker S'

    A timeserie of prices and volume that you would get placing an order in that moment.
    :return [{'o': float, 'v': float, 't': datetime]: (timestamp, price you would get, volume you would get)
    """
    ticks = get_sticker_odds(conn, sticker)
    rets = get_volume_at_price(ticks, limit_price, volume, is_back)
    return rets


def get_price_for_volume_at(conn, sticker, limit_price, volume, is_back, timestamp):
    """

    'Which price and volume would I get for V volume, P price on sticker S'

    prices and volume that you would get placing an order in that moment.
    :return [{'o': float, 'v': float, 't': datetime]: (timestamp, price you would get, volume you would get)
    """
    tick = get_last_tick_before(conn, sticker, timestamp)
    rets = get_volume_at_price([tick], limit_price, volume, is_back)
    return rets[0]


if __name__ == '__main__':
    conn = get_connection('tennis')
    # print get_sticker(conn, 'T-EENP1923578-FT12-A.BF')

    print get_matched_volume_between(
        conn, 'T-EENP2191253-FT12-A.BF',
        datetime(2016, 2, 22, 11, 25, 0, 000, tzinfo=pytz.utc),
        datetime(2016, 2, 22, 11, 48, 0, 000, tzinfo=pytz.utc)
    )
    print get_price_for_volume(
        conn, 'T-EENP2191253-FT12-A.BF',
        1.5, 1000, True)
