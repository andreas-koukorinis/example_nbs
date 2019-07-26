import logging

from sgmtradingcore.analytics.abc.abc import get_connection
from sgmtradingcore.analytics.abc.odds import get_first_in_play_tick_from_market
from sgmtradingcore.analytics.abc.stickers import stickers_from_meta
from sgmtradingcore.strategies.abc_db.model import MarketTick as MT


def get_kick_off_from_market_data(conn, event_id, restrict_to=None):
    """
    Check the attribute dbl (deadball) in the market data and takes the first
    where it is False. it checks all the stickers (slow) unless you restrict
    the list of stickers thanks to the argument restrict_to

    Parameters
    ----------
    event_id: (str) gsm_id e.g "GSM1234567"
    conn: (MongoPersister) mongo connection
    restrict_to: (list of str) look up stickers which contain one of the
        str from the list.

    Returns
    -------
    (datetime) or None
    """
    # Get list of stickers in the metadata and then queries
    # some of them one by one and return the minimum timestamp where dbl is
    # False from a range of markets
    stickers = stickers_from_meta(conn, event_id, restrict_to=restrict_to)
    if len(stickers) == 0:
        logging.debug('No stickers restricted to {} available for event_id in'
                      'metadata'.format(restrict_to))
        return None

    kick_off = get_kick_off_from_stickers_market_data(conn, stickers)
    return kick_off


def get_kick_off_from_stickers_market_data(conn, stickers):
    kick_off = None
    for stkr in stickers:
        tk = get_first_in_play_tick_from_market(conn, stkr)
        if tk is None:
            logging.error('Sticker {} not found in inplay market data'.format(stkr))
            continue
        stkr_first_ip = tk[MT.timestamp.db_field]
        if kick_off is None or stkr_first_ip < kick_off:
            kick_off = stkr_first_ip
    return kick_off


if __name__ == '__main__':
    conn = get_connection('tennis')
    print get_kick_off_from_market_data(conn, 'ENP2191253')
