from sgmtradingcore.analytics.abc.abc import get_connection


def stickers_from_meta(conn, event_id, restrict_to=None):
    stickers = existing_stickers_from_meta(conn, event_id)
    if restrict_to is not None:
        stickers = [s for s in stickers
                    if any([sub_s in s for sub_s in restrict_to])]
    return stickers


def existing_stickers_from_meta(conn, event_id):
    """
    Get the stickers where we are supposed to have market data according
    to the fixture metadata collection

    Parameters
    ----------
    conn: (MongoPersister) open connection to the abc_db database
    event_id: (str) for example "GSM1234567"

    Returns
    -------
    The stickers found in the fixture metadata collection
    """
    query = {'eid': event_id}
    proj = {'stkrs': 1}
    coll = 'fixture_meta' if conn.sport == 'football' else conn.sport+'_fixture_meta'
    res = [f for f in conn[coll].find(query, proj)]

    if len(res) == 0:
        return []
    else:
        return res[0]['stkrs']


def existing_stickers_from_market(conn, event_id):
    """
    Get the stickers that have market data

    Parameters
    ----------
    conn: (MongoPersister) open connection to the abc_db database
    event_id: (str) for example "GSM1234567"

    Returns
    -------
    [str]
    """
    filter = {'stkr.eid': event_id}
    coll = conn.sport + '_market_data'
    res = [f for f in conn[coll].distinct('stkr.str', filter=filter)]
    return res


if __name__ == '__main__':
    conn = get_connection('tennis')
    # print get_sticker_odds(conn, 'T-EENP1923578-FT12-A.BF')
    print existing_stickers_from_meta(conn, 'ENP1923578')
    print existing_stickers_from_market(conn, 'ENP1923578')
