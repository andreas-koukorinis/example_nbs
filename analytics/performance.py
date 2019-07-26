from collections import defaultdict
from datetime import timedelta
from sgmtradingcore.core.trading_types import BetOutcome, IntEnum, OrderStatus
from stratagemdataprocessing import data_api
from stratagemdataprocessing.enums.markets import TennisMarkets, TennisSelections
from stratagemdataprocessing.parsing.common.stickers import parse_sticker

import pandas as pd


def convert_bet_states_to_array(order_states):
    """
    Convert bet_states DataFrame to array of bets containing only last known bet state
    :param order_states: DataFrame of bet states
    :return: array of bet objects
    """
    orders = dict()
    for timestamp, order in order_states.iteritems():
        _, market_scope, market, params, _ = parse_sticker(order.sticker)
        orders[order.id] = {
            'event': market_scope[1],
            'pnl': order.pnl,
            'market': market,
            'commission': order.commission,
            'stake': order.size,
            'is_back': order.is_back,
            'odds': order.price if order.matched_odds == 0 else order.matched_odds,
            'date': timestamp,
        }
    return orders.values()


def name_from_class_var(cls, var):
    if issubclass(cls, IntEnum):
        return [k for k, v in vars(cls)['_member_map_'].iteritems() if v == var][0]
    else:
        return [k for k, v in vars(cls).iteritems() if v == var][0]


def dict_from_bet_state(dt, bs):

    out = {
        'fixture_id': bs.event,
        'dt': dt,
        'sticker': bs.sticker,
        'market_id': bs.market,
        'selection_id': bs.params[0],
        'bet_status': bs.status,
        'is_back': bs.is_back, 'outcome': name_from_class_var(BetOutcome, bs.outcome),
        'pnl': float(bs.pnl), 'limit_order_odds': float(bs.price),
        'liability': float(bs.details.get('liability', -1.)),
        'stake': float(bs.size),
        'odds': bs.matched_odds
    }

    details = {key: bs.details[key] for key in bs.details}
    out.update(details)
    return out


def apply_betfair_commission(orders, commission=0.035):
    # Calculate aggregate stakes and pnls for 'event'.'market' groups
    pnl = defaultdict(lambda: 0.0)
    stakes = defaultdict(lambda: 0.0)
    for order in orders:
        pnl[(order['event'], order['market'])] += order['pnl']
        if order['pnl'] > 0:
            stakes[(order['event'], order['market'])] += order['stake']

    # Now add weighted commissions to winning orders only and only in case that market has positive PnL
    for order in orders:
        if order['pnl'] > 0 and pnl[(order['event'], order['market'])] > 0:
            order['commission'] = (commission * pnl[(order['event'], order['market'])]) * \
                                  (order['stake'] / stakes[(order['event'], order['market'])])
        else:
            order['commission'] = 0

    return orders


def get_trading_days(sport, start_dt, end_dt, sport_filter):
    """
    Get the number and dates of the trading days in the given period for the given filter.
    The rules are the following:

    - If the period between the start and end date spans multiple business years, use
      the total trading days of the year the period mostly falls in
    - If the period between the start and end date falls within previous business years,
      use the trading days of the latest full year
    - If the period between the start and end date falls in the current year, get the trading
      dates of the current year but use the total number of trading days from the previous year

    The above is implemented in the data api so we just call through
    """

    trading_days, total = data_api.get_trading_days(
        sport, start_dt, end_dt, sport_filter.to_filter())

    return trading_days, total


def print_backtest_stats(bet_states, start_date, end_date):
    bets = {}

    for bet_state in bet_states:
        bets[bet_state.id] = bet_state

    # Print bets grouped by match
    gsm_ids = []
    while True:
        gsm_id = None
        for bet_id in bets:
            if gsm_id is None and bets[bet_id].event not in gsm_ids:
                gsm_id = bets[bet_id].event
                match_pnl = 0
                gsm_ids.append(gsm_id)
                print '---- %s ----' % gsm_id
            if gsm_id == bets[bet_id].event:
                match_pnl += bets[bet_id].pnl
                print "%s\t%.3f\t%s" % (bets[bet_id].outcome, bets[bet_id].pnl, bets[bet_id])
        if gsm_id is None:
            print '--------------------'
            break
        else:
            print 'Match PNL: %.3f' % match_pnl

    pnl = sum(bs.pnl for bs in bets.values())
    commission = sum(bs.commission for bs in bets.values())
    start_date = start_date.date()
    end_date = end_date.date()
    bet_no = len(bets)
    trading_days = []

    trading_day = start_date
    while trading_day <= end_date:
        trading_days.append(trading_day)
        trading_day += timedelta(days=1)

    bet_counter = defaultdict(lambda: 0)
    for bet in bets.values():
        bet_counter[bet.outcome] += 1

    trading_days_no = len(trading_days)

    if bet_counter[BetOutcome.LOSS] > 0:
        bet_wl_ratio = bet_counter[BetOutcome.WIN] / (bet_counter[BetOutcome.LOSS] + 0.0)
    elif bet_counter[BetOutcome.WIN] > 0:
        bet_wl_ratio = '+inf'
    else:
        bet_wl_ratio = 'N/A'

    print 'start_date:       %s' % start_date
    print 'end_date:         %s' % end_date
    print 'trading_days_no:  %s' % trading_days_no
    print 'bets_no:          %s' % bet_no
    print 'bet_wins:         %s' % bet_counter[BetOutcome.WIN]
    print 'bet_losses:       %s' % bet_counter[BetOutcome.LOSS]
    print 'bet_unknown:      %s' % bet_counter[BetOutcome.UNKNOWN]
    print 'bet_wl_ratio:     %s' % bet_wl_ratio
    print 'pnl:              %s' % pnl
    print 'commission:       %s' % commission
    print '--------------------'

    info = {'bets_no': bet_no, 'bet_wins': bet_counter[BetOutcome.WIN], 'bet_losses': bet_counter[BetOutcome.LOSS],
            'bet_unknown': bet_counter[BetOutcome.UNKNOWN], 'bet_wl_ratio': bet_wl_ratio, 'pnl': pnl,
            'commission': commission}

    return info


def backtest_stats_forcsv(bet_infos):
    bets = {}
    data_df = pd.DataFrame(columns=['event', 'outcome', 'status', 'market', 'selection', 'b/l', 'stake', 'gross_odds',
                                    'net_odds', 'bookmaker', 'pnl', 'arb_id'])
    i = 0

    for bet_info in bet_infos:
        bets[bet_info.id] = bet_info

    # add bet stats to df
    gsm_ids = []
    while True:
        gsm_id = None
        for bet_id in bets:
            if gsm_id is None and bets[bet_id].event not in gsm_ids:
                gsm_id = bets[bet_id].event
                gsm_ids.append(gsm_id)
            if gsm_id == bets[bet_id].event:
                data_df.ix[str(i), 'event'] = bets[bet_id].event
                data_df.ix[str(i), 'outcome'] = \
                    vars(BetOutcome).get('_value2member_map_').get(bets[bet_id].outcome).name
                data_df.ix[str(i), 'status'] = \
                    vars(OrderStatus).get('_value2member_map_').get(bets[bet_id].status).name
                data_df.ix[str(i), 'market'] = \
                    [key for key, value in vars(TennisMarkets).iteritems() if value == bets[bet_id].market][0]
                data_df.ix[str(i), 'selection'] = \
                    [key for key, value in vars(TennisSelections).iteritems() if value == bets[bet_id].params[0]][0]
                data_df.ix[str(i), 'b/l'] = bets[bet_id].is_back
                data_df.ix[str(i), 'stake'] = bets[bet_id].size
                data_df.ix[str(i), 'gross_odds'] = bets[bet_id].matched_odds
                data_df.ix[str(i), 'net_odds'] = bets[bet_id].price
                data_df.ix[str(i), 'bookmaker'] = bets[bet_id].execution_details['bookmaker']
                exchange_rate = bets[bet_id].exchange_rate
                if exchange_rate == 0:
                    exchange_rate = 1
                data_df.ix[str(i), 'pnl'] = bets[bet_id].pnl/exchange_rate
                data_df.ix[str(i), 'arb_id'] = bets[bet_id].details['arb_id']
                data_df.ix[str(i), 'arb_type'] = bets[bet_id].details['arb_type']
                i += 1

        if gsm_id is None:
            print '--------------------'
            break

    # calculate net profit
    for row, col in data_df.iterrows():
        if data_df.ix[row, 'outcome'] == 'LOSS':
            if data_df.ix[row, 'b/l']:
                data_df.ix[row, 'net_profit'] = -1 * data_df.ix[row, 'stake']
            elif not data_df.ix[row, 'b/l']:
                data_df.ix[row, 'net_profit'] = -1 * data_df.ix[row, 'stake'] * (data_df.ix[row, 'net_odds'] - 1)
        elif data_df.ix[row, 'outcome'] == 'WIN':
            if data_df.ix[row, 'b/l']:
                data_df.ix[row, 'net_profit'] = (data_df.ix[row, 'net_odds'] - 1) * data_df.ix[row, 'stake']
            elif not data_df.ix[row, 'b/l']:
                data_df.ix[row, 'net_profit'] = data_df.ix[row, 'stake']

    # fill net_profit with -inf where outcome is unknown
    data_df = data_df.fillna(value=float('-Inf'))

    # calculate net_profit per arb
    for row, column in data_df.iterrows():
        data_df.ix[row, 'pnl'] = round(
            data_df.loc[data_df['arb_id'] == data_df.ix[row, 'arb_id']].sum()['net_profit'], 2)

    # set arb_id and event as indices
    data_df.set_index(['arb_id', 'event'], inplace=True)

    return data_df
