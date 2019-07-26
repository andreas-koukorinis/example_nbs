import ciso8601
import pytz
from collections import defaultdict

from datetime import timedelta

from sgmtradingcore.analytics.execution.benchmarks import exchange_vwap, bookmaker_twabp
from stratagemdataprocessing.bookmakers.common.odds.cache import HistoricalOddsCache
from stratagemdataprocessing.data_api import get_settled_orders, get_instructions_by_id, get_match_info
from stratagemdataprocessing.dbutils.mysql import MySQLClient
from stratagemdataprocessing.enums.odds import Bookmakers
from stratagemdataprocessing.parsing.common.stickers import parse_sticker


_SQL_QUERY = """
SELECT
    IFNULL(e.id, "-1") AS "eventID",
    IFNULL(p.id,"-1") AS "playerID",
    IFNULL(ep.number,"-1") AS "participantNumber",
    IFNULL(results.result_code, "-") AS "resultCode",
    IFNULL(results.value, -1) AS "resultValue",
    results.ut as "actionDate",
    results.n as "n"
FROM
    result AS results INNER JOIN
    event_participants AS ep ON results.event_participantsFK = ep.id INNER JOIN
    participant AS p ON ep.participantFK = p.id INNER JOIN
    event AS e ON ep.eventFK = e.id
WHERE
    e.id in (%s)
ORDER BY eventID, actionDate ASC, resultCode ASC
"""


def get_set_end_times(event_action_data):

    result = defaultdict(dict)

    for e in event_action_data:
        event_id = 'ENP%s' % e['eventID']
        result_code = e['resultCode']
        if result_code in ('set1', 'set2', 'set3'):
            result[event_id][result_code] = e['actionDate']

    return result


def execution_data_tennis_sip(trading_user_id, start_dt, end_dt, strategy_id, strategy_run_id=None):

    cache = HistoricalOddsCache()
    mysql_enet = MySQLClient.init_from_config(auto_connect=True)

    orders = get_settled_orders(trading_user_id, start_dt, end_dt, strategy_id, strategy_run_id)

    instruction_ids = {order['instruction_id'] for order in orders if 'instruction_id' in order}
    instructions = get_instructions_by_id(trading_user_id, list(instruction_ids))

    stickers = {order['sticker'] for order in orders if len(order['sticker']) > 0}
    enp_ids = set()

    for sticker in stickers:
        sport, (_, event_id), market, params, _ = parse_sticker(sticker)
        enp_ids.add(event_id[3:])

    if len(enp_ids) == 0:
        return []

    results_query = _SQL_QUERY % ', '.join(enp_ids)
    results_tz = pytz.timezone('Europe/London')

    try:
        event_action_data = mysql_enet.select(results_query, as_list=True)
    except:
        print results_query
        raise

    set_end_times = get_set_end_times(event_action_data)
    rows = []

    for order in orders:
        sticker = order['sticker']
        rate = order['exchange_rate']

        if len(sticker) == 0:
            continue

        sport, (_, event_id), market, params, _ = parse_sticker(sticker)

        start_dt_end_set1 = results_tz.localize(set_end_times[event_id]['set1'])
        start_dt_end_set2 = results_tz.localize(set_end_times[event_id]['set2'])

        placed_dt = ciso8601.parse_datetime(order['placed_time'])
        if start_dt_end_set1 - timedelta(minutes=5) < placed_dt < start_dt_end_set2:
            bf_vwap = exchange_vwap(
                sticker, Bookmakers.BETFAIR, start_dt_end_set1, start_dt_end_set2, odds_cache=cache, return_nan=True)
            suffix = 'set2'
        else:
            if 'set3' not in set_end_times[event_id]:
                bf_vwap = float('nan')
                suffix = 'unknown'
            else:
                start_dt_end_set3 = results_tz.localize(set_end_times[event_id]['set3'])
                if start_dt_end_set2 - timedelta(minutes=5) < placed_dt < start_dt_end_set3:
                    bf_vwap = exchange_vwap(
                        sticker, Bookmakers.BETFAIR, start_dt_end_set2, start_dt_end_set3, odds_cache=cache, return_nan=True)
                    suffix = 'set3'
                else:
                    bf_vwap = float('nan')
                    suffix = 'unknown'

        try:
            rows.append({
                'order_id': order['id'],
                'order_source': order['source'],
                'instruction_id': order.get('instruction_id', ''),
                'strategy_descr': order['strategy_descr'],
                'sticker': sticker,
                'order_size': order['size'] / rate,
                'source': order['source'],
                'matched_size': order['size_matched'] / rate,
                'pnl': order['outcome']['net'] / rate,
                'limit_price': order['price'],
                'average_price': order['average_price_matched'],
                'side': order['bet_side'],
                'bf_vwap_%s' % suffix: bf_vwap,
                'bookmaker': order['execution_details'].get('bookmaker', ''),
                'benchmark_price': bf_vwap,
            })
        except:
            print order
            raise

    return rows