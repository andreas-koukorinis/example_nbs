from stratagemdataprocessing.enums.odds import SelectionStatus
from stratagemdataprocessing.dbutils.mongo import MongoPersister

ACTIVE_STATUS = SelectionStatus.to_str_symbol(SelectionStatus.ACTIVE)


def get_connection(sport):
    if sport == 'football':
        conn = MongoPersister.init_from_config('abc_football_v2', auto_connect=True)
    elif sport == 'tennis':
        conn = MongoPersister.init_from_config('abc_tennis_v2', auto_connect=True)
    elif sport == 'basketball':
        conn = MongoPersister.init_from_config('abc_basketball_v2', auto_connect=True)
    else:
        raise ValueError("Bad sport")
    conn.sport = sport
    return conn
