import logging
from calendar import monthrange, month_name
from datetime import datetime, timedelta

import pandas as pd
from jinja2 import Template

from sgmtradingcore.analytics.commissions import get_commission_rates_for_user
from sgmtradingcore.analytics.execution.tennis_sip import execution_data_tennis_sip
from sgmtradingcore.core.notifications import send_trading_system_email
from sgmtradingcore.strategies.config.configurations import TRADING_USER_MAP

_DATA_GETTERS = {
    'tennis_sip_ATP': execution_data_tennis_sip,
    'tennis_sip_WTA': execution_data_tennis_sip,
    'tennis_sip_v2_ATP': execution_data_tennis_sip,
    'tennis_sip_v2_WTA': execution_data_tennis_sip,
}


def apply_commission(row, commissions):
    if row['bookmaker'] not in commissions:
        logging.warn('Do not know the commission for bookmaker %s for order id %s. Defaulting to 0%%' % (
            row['order_id'], row['bookmaker']))
        return row['average_price']

    if row['side'] == 'back':
        net_price = 1 + (row['average_price'] - 1) * (1 - commissions[row['bookmaker']])
    else:
        net_price = 1 + (row['average_price'] - 1) / (1 - commissions[row['bookmaker']])

    return net_price


def apply_slippage_pnl(row):
    if row['outcome'] == 1 and row['side'] == 'back':
        return row['matched_size'] * row['slippage_benchmark']
    elif row['outcome'] == -1 and row['side'] == 'lay':
        return row['matched_size'] * row['slippage_benchmark']
    else:
        return 0


def apply_slippage_benchmark(row):
    if row['side'] == 'back':
        return row['average_price_net'] - row['benchmark_price']
    elif row['side'] == 'lay':
        return row['benchmark_price'] - row['average_price_net']
    else:
        raise ValueError('Invalid side %s' % row['side'])


_commissions = dict()


def execution_report(raw_data, trading_user_id):
    df_all = pd.DataFrame(raw_data)

    if len(df_all) == 0:
        return {}
    global _commissions
    if trading_user_id not in _commissions:
        _commissions[trading_user_id] = get_commission_rates_for_user(trading_user_id)

    df_all['average_price_net'] = df_all.apply(lambda row: apply_commission(
        row, commissions=_commissions[trading_user_id]), axis=1)
    df_all['slippage_benchmark'] = df_all.apply(lambda row: apply_slippage_benchmark(row), axis=1)
    df_all['outcome'] = df_all.apply(lambda row: 1 if row['pnl'] > 0 else -1 if row['pnl'] < 0 else 0, axis=1)
    df_all['slippage_benchmark_size'] = df_all.apply(
        lambda row: row['slippage_benchmark'] * row['matched_size'], axis=1)
    df_all['slippage_pnl'] = df_all.apply(lambda row: apply_slippage_pnl(row), axis=1)
    df_all['source'] = df_all.apply(
        lambda row: 'auto' if row['source'] == 'trading_system' else 'manual', axis=1
    )

    total_matched_size = df_all['matched_size'].sum()
    total_pnl = df_all['pnl'].sum()
    slippage_benchmark_all = df_all['slippage_benchmark_size'].sum() / df_all['matched_size'].sum()
    slippage_pnl_benchmark_all = df_all['slippage_pnl'].sum()

    slippage_benchmark_source = df_all[['slippage_benchmark_size', 'matched_size', 'source']].groupby(['source']).sum()
    slippage_benchmark_source['slippage'] = slippage_benchmark_source.apply(
        lambda row: row['slippage_benchmark_size'] / row['matched_size'], axis=1)

    results_row = {
        'slip_odds': slippage_benchmark_all,
        'slip_pnl': slippage_pnl_benchmark_all,
        'total_stake': total_matched_size,
        'total_pnl': total_pnl,
    }

    for source, row in slippage_benchmark_source.iterrows():
        results_row['slip_%s_odds' % source] = row['slippage']
        results_row['total_%s_stake' % source] = row['matched_size']

    results_row.setdefault('slip_manual_odds', 0)
    results_row.setdefault('total_manual_stake', 0)

    return results_row


def month_year_iter(start_month, start_year, end_month, end_year):
    ym_start = 12 * start_year + start_month - 1
    ym_end = 12 * end_year + end_month - 1
    for ym in range(ym_start, ym_end):
        y, m = divmod(ym, 12)
        yield y, m+1


def execution_report_main(trading_user_id, strategy_ids, start_date, end_date, pivot, email_to):

    template_results = []
    columns = ['month', 'year', 'slip_auto_odds', 'total_auto_stake', 'slip_manual_odds',
               'total_manual_stake', 'slip_pnl', 'total_pnl', 'total_stake']

    if pivot == 'daily':
        columns.insert(0, 'day')

        for strategy_id in strategy_ids:
            results = []

            current_date = start_date
            while current_date <= end_date:

                start_dt = datetime(current_date.year, current_date.month, current_date.day, 0, 0, 0)
                end_dt = datetime(current_date.year, current_date.month, current_date.day, 23, 59, 59)

                data_getter = _DATA_GETTERS[strategy_id]
                raw_data = data_getter(trading_user_id, start_dt, end_dt, strategy_id, None)

                results_row = execution_report(raw_data, trading_user_id)

                current_date += timedelta(days=1)
                if len(results_row) == 0:
                    continue

                results_row['day'] = current_date.day
                results_row['month'] = month_name[current_date.month]
                results_row['year'] = current_date.year
                results.append(results_row)

            if len(results):
                results_df = pd.DataFrame(results)[columns]

                template_results.append({'strategy_id': strategy_id, 'report_html': results_df.to_html()})

    elif pivot == 'monthly':
        for strategy_id in strategy_ids:
            results = []

            for y, m in month_year_iter(start_date.month, start_date.year, end_date.month, end_date.year):
                mr = monthrange(y, m)

                start_dt = datetime(y, m, 1, 0, 0, 0)
                end_dt = datetime(y, m, mr[1], 23, 59, 59)

                data_getter = _DATA_GETTERS[strategy_id]
                raw_data = data_getter(trading_user_id, start_dt, end_dt, strategy_id, None)

                results_row = execution_report(raw_data, trading_user_id)

                if len(results_row) == 0:
                    continue

                results_row['month'] = month_name[m]
                results_row['year'] = y
                results.append(results_row)

            if len(results):
                results_df = pd.DataFrame(results)[columns]

                template_results.append({'strategy_id': strategy_id, 'report_html': results_df.to_html()})

    else:
        raise ValueError('Bad pivot type %s' % pivot)

    template = Template("""
Execution quality report for {{ trading_user }}
<br/>
{% for row in rows %}
{{ row.strategy_id }}<br/>
{{ row.report_html }}<br/>
{% endfor %}

""")

    subject = 'Execution performance report for %s for user %s' % (
        ', '.join(strategy_ids), TRADING_USER_MAP[trading_user_id])

    html_ = template.render(rows=template_results, trading_user=TRADING_USER_MAP[trading_user_id])
    send_trading_system_email('', html_, subject, email_to)
