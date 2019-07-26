import datetime as dt
import logging
import tempfile
import time
import traceback
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template
from stratagemdataprocessing.data_api import get_capital_timeseries
from stratagemdataprocessing.parsing.common.stickers import sticker_parts_from_sticker, parse_sticker
from tabulate import tabulate
from sgmtradingcore.backtesting.backtest import FrameworkHistoricalProviders
import sgmtradingcore.backtesting.persistence as persistence
from sgmtradingcore.backtesting.automatic_backtest import TestResult, split_by_time_period, _get_size_matched, \
    _get_size, get_strategy_code_from_instruction, _get_orders_debug_print
from sgmtradingcore.backtesting.backtest_runner import run_backtest_main
from sgmtradingcore.core.notifications import send_trading_system_email
from sgmtradingcore.core.trading_types import OrderStatus
from sgmtradingcore.strategies.config.configurations import TRADING_USER_MAP
from sgmtradingcore.strategies.realtime import StrategyFactory
from sgmtradingcore.strategies.strategy_base import StrategyStyle
from sgmtradingcore.analytics.comparison.trades_stats import StrategyRunStatsHelper


def _send_email_retry(text, html_, subject, to_, attachments=None):
    try:
        send_trading_system_email(text, html_, subject, to_, files=attachments)
    except:
        time.sleep(4)
        send_trading_system_email(text, html_, subject, to_, files=attachments)


def _get_order_statuses(comparison_input, tmp_dir):
    values = [(comparison_input.name1, comparison_input.all_orders1),
              (comparison_input.name2, comparison_input.all_orders2)]
    file_paths = []
    message = ""
    for name, orders in values:
        message += "Tot {} {} orders\n".format(name, len(orders))

        per_bookmaker_count = {}
        orders_by_bookmaker_and_status = dict()
        for o in orders:
            if 'provider' in o['execution_details']:
                bm = "{}_{}".format(o['execution_details']['provider'], o['execution_details']['bookmaker'])
            else:
                bm = "{}".format(o['execution_details']['bookmaker'])
            if bm not in orders_by_bookmaker_and_status:
                orders_by_bookmaker_and_status[bm] = {OrderStatus.REJECTED: 0,
                                                      OrderStatus.FAILED: 0,
                                                      OrderStatus.CANCELLED: 0,
                                                      OrderStatus.SETTLED: 0}
            if OrderStatus(o['status']) not in orders_by_bookmaker_and_status[bm]:
                orders_by_bookmaker_and_status[bm][OrderStatus(o['status'])] = 0
            orders_by_bookmaker_and_status[bm][OrderStatus(o['status'])] += 1

            if bm not in per_bookmaker_count:
                per_bookmaker_count[bm] = 0
            per_bookmaker_count[bm] += 1

        headers = sorted([bm for bm, _ in orders_by_bookmaker_and_status.iteritems()])
        data = []
        for s in OrderStatus:
            line = list()
            line.append(str(s))
            for bm in headers:
                statuses = orders_by_bookmaker_and_status[bm]
                if s in statuses:
                    line.append(statuses[s])
                else:
                    line.append(0)
            data.append(line)
        message += tabulate(data, headers=headers)

        labels = sorted(orders_by_bookmaker_and_status.keys())
        values_rejected_perc = [float(orders_by_bookmaker_and_status[bm][OrderStatus.REJECTED]) / per_bookmaker_count[bm]
                                for bm in labels]
        values_failed_perc = [float(orders_by_bookmaker_and_status[bm][OrderStatus.FAILED]) / per_bookmaker_count[bm]
                              for bm in labels]
        values_settled_perc = [float(orders_by_bookmaker_and_status[bm][OrderStatus.SETTLED]) / per_bookmaker_count[bm]
                               for bm in labels]
        values_cancelled_perc = [float(orders_by_bookmaker_and_status[bm][OrderStatus.CANCELLED]) /
                                 per_bookmaker_count[bm] for bm in labels]
        if not len(values_rejected_perc):
            values_rejected_perc = [0.0]
        if not len(values_failed_perc):
            values_failed_perc = [0.0]
        if not len(values_settled_perc):
            values_settled_perc = [0.0]
        if not len(values_cancelled_perc):
            values_cancelled_perc = [0.0]

        plt.bar(range(max(len(labels), 1)),
                values_rejected_perc,
                width=0.3,
                label='% Rejected',
                color='y')
        plt.bar(range(max(len(labels), 1)),
                values_failed_perc,
                width=0.3,
                bottom=values_rejected_perc,
                label='% Failed',
                color='r')
        plt.bar(range(max(len(labels), 1)),
                values_settled_perc,
                width=0.3,
                bottom=[x + y for x, y in zip(values_rejected_perc, values_failed_perc)],
                label='% Settled',
                color='b'
                )
        plt.bar(range(max(len(labels), 1)),
                values_cancelled_perc,
                width=0.3,
                bottom=[x + y + z for x, y, z in zip(values_rejected_perc, values_failed_perc, values_settled_perc)],
                label='% Cancelled',
                color='g'
                )

        # plt.axis(labels)

        plt.tight_layout(pad=3)
        # plt.gcf().subplots_adjust(bottom=0.15)
        plt.xticks(range(len(labels)), labels, rotation=-90)
        plt.legend()

        plt.title('Statuses {}'.format(name))
        file_paths.append('%s/%s' % (tmp_dir, "statuses_{}.png".format(name)))
        plt.savefig(file_paths[-1])
        plt.close()
        message += "\n\n"

    res = TestResult(name="Orders status count",
                     good_report_message=message,
                     attachments=file_paths,
                     )
    res.success = False
    return res


def _get_order_sources(comparison_input):
    message = "{} orders source\n".format(comparison_input.name1)
    headers = set()
    for o in comparison_input.all_orders1:
        source = o['source'] if 'source' in o else 'unknown'
        headers.add(source)
    headers = sorted(headers)
    lines = []

    for strategy in comparison_input.strategies:
        orders_count_by_source = dict()
        for o in strategy.orders1:
            source = o['source'] if 'source' in o else 'unknown'
            if source not in orders_count_by_source:
                orders_count_by_source[source] = 0
            orders_count_by_source[source] += 1

        line = [strategy.get_short_name()]
        for source in headers:
            if source not in orders_count_by_source:
                line.append("0")
            else:
                line.append("{}".format(orders_count_by_source[source]))
        lines.append(line)

    message += tabulate(lines, headers=headers)
    res = TestResult(name="Orders sources",
                     good_report_message=message,
                     )
    res.success = False
    return res


def _get_pnl_distrib_by_period(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = ""
    attachments = []
    for strategy in comparison_input.strategies:
        periods_prod = strategy.daily_periods_1
        periods_back = strategy.daily_periods_2
        pnls_back = [p.get_pnl() for p in periods_back if p.has_orders()]
        pnls_prod = [p.get_pnl() for p in periods_prod if p.has_orders()]
        # bins = np.linspace(min(pnls_prod+pnls_back+[0]), max(pnls_prod+pnls_back+[0]), len(periods_prod))
        bins = min(max(len(periods_back), len(periods_prod)), 50)
        # hist = hist*1.0/sum(hist)  # normalize
        pnl_range = (min(pnls_back + pnls_prod + [0]), max(pnls_back + pnls_prod + [0]))
        if len(pnls_prod):
            plt.hist(pnls_prod,
                     rwidth=1.0,  # size of bars relative to the bins
                     bins=bins,
                     label='Pnl distrib {} {}/{} days'.format(
                         comparison_input.name1, len(pnls_prod), len(periods_prod)),
                     color='b',
                     range=pnl_range,
                     alpha=0.5)
        if len(pnls_back):
            plt.hist(pnls_back,
                     rwidth=1.0,
                     bins=bins,
                     label='Pnl distrib {} {}/{} days'.format(
                         comparison_input.name2, len(pnls_back), len(periods_back)),
                     color='y',
                     range=pnl_range,
                     alpha=0.5)
        # plt.axis(labels)

        if len(pnls_prod) or len(pnls_back):
            plt.tight_layout(pad=3)
            # plt.gcf().subplots_adjust(bottom=0.15)
            # plt.xticks(range(len(labels)), labels, rotation=-90)
            plt.legend()
            plt.title('Daily pnl {}'.format(strategy.get_short_name()))

            file_path = '%s/%s' % (tmp_dir, "pnl_distrib_{}.png".format(strategy.get_short_name()))
            plt.savefig(file_path)
            plt.close()
            attachments.append(file_path)

    res = TestResult(name="Strategy PNL {}/{}".format(comparison_input.name1, comparison_input.name2),
                     good_report_message=message,
                     attachments=attachments,
                     )
    res.success = True
    return res


def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + dt.timedelta(n)


def _get_main_stats(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """
    Create main stats

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    headers = ["Orders", "Settled", "Matched vol",
               "Avg daily ROI", "Tot PNL", "PNL Diff",
               "Avg daily Capital"]
    lines = []
    message = ""
    attachments = []
    for strategy in comparison_input.strategies:
        daily_periods_prod = strategy.daily_periods_1
        daily_periods_back = strategy.daily_periods_2

        # Total ROI
        prod_daily_rois = []
        pnl_prod = 0.0
        for p in daily_periods_prod:
            date = p.start_datetime.date()
            capital_allocation = strategy.capitals_serie
            if isinstance(strategy.capitals_serie, dict):
                capital_allocation = strategy.capitals_serie[date] if date in strategy.capitals_serie else 0.
            pnl = p.get_pnl()
            pnl_prod += pnl
            if capital_allocation:
                roi = 100.0 * pnl / capital_allocation
                prod_daily_rois.append(roi)

        back_daily_rois = []
        pnl_back = 0.0
        for p in daily_periods_back:
            date = p.start_datetime.date()
            capital_allocation = strategy.capitals_serie
            if isinstance(strategy.capitals_serie, dict):
                capital_allocation = strategy.capitals_serie[date] if date in strategy.capitals_serie else 0.
            pnl = p.get_pnl()
            pnl_back += pnl
            if capital_allocation > 0.0:
                roi = 100.0 * pnl / capital_allocation
                back_daily_rois.append(roi)

        capitals = [c for c in strategy.capitals_serie.values() if c > 0.0] if isinstance(strategy.capitals_serie, dict) \
            else [strategy.capitals_serie]
        avg_capital = int(sum(capitals) / len(capitals)) if len(capitals) else 0

        line = [strategy.get_short_name(),
                "{:5.0f}/{:>5.0f}".format(len(strategy.orders1), len(strategy.orders2)),
                "{:5.0f}/{:>5.0f}".format(len([o for o in strategy.orders1 if o['status'] == OrderStatus.SETTLED]),
                                          len([o for o in strategy.orders2 if o['status'] == OrderStatus.SETTLED])),
                "{:8.0f}/{:>8.0f}".format(  # Matched vol
                    sum([o['size_matched'] for o in strategy.orders1 if o['status'] == OrderStatus.SETTLED]),
                    sum([o['size_matched'] for o in strategy.orders2 if o['status'] == OrderStatus.SETTLED])),
                "{:5.2f}/{:>5.2f}%".format(sum(prod_daily_rois) / len(prod_daily_rois) if len(prod_daily_rois) else 0.0,
                                           sum(back_daily_rois) / len(back_daily_rois) if len(back_daily_rois) else 0.0
                                           ),
                "{:6.0f}/{:>6.0f}".format(pnl_prod, pnl_back),
                "{:5.1f}%".format((pnl_back - pnl_prod) * 100 / abs(pnl_prod) if pnl_prod != 0. else 9999),
                "{}".format(avg_capital)
                ]
        lines.append(line)

    message += tabulate(lines, headers=headers, stralign='right')

    res = TestResult(name="ROI {}/{}".format(comparison_input.name1, comparison_input.name2),
                     good_report_message=message,
                     attachments=attachments,
                     )
    res.success = True
    return res


def _get_weekly_roi(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """
    Create weekly ROI chart and ROI ZScore chart

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    headers = ["Orders", "Settled", "Avg daily ROI", "Tot PNL", "PNL Diff", "Avg daily Capital"]
    lines = []
    message = ""
    attachments = []
    for strategy in comparison_input.strategies:
        # Weekly ROI
        weekly_periods_prod = split_by_time_period(strategy.orders1, 7,
                                                   start_datetime.date(),
                                                   end_datetime.date(),
                                                   use_cache=use_cache)
        weekly_periods_back = split_by_time_period(strategy.orders2, 7,
                                                   start_datetime.date(),
                                                   end_datetime.date(),
                                                   use_cache=use_cache)

        weekly_roi_prod = []
        for p in weekly_periods_prod:
            daily_rois = []
            pnls = p.get_daily_pnl()
            for date in _daterange(p.start_datetime.date(), p.end_datetime.date()):
                capital_allocation = strategy.capitals_serie
                if isinstance(strategy.capitals_serie, dict):
                    capital_allocation = strategy.capitals_serie[date] if date in strategy.capitals_serie else 0.
                pnl = pnls[date]
                roi = 100.0 * pnl / capital_allocation if capital_allocation else 0.0
                daily_rois.append(roi)

            weekly_roi_prod.append(sum(daily_rois) / len(daily_rois))

        weekly_roi_back = []
        for p in weekly_periods_back:
            daily_rois = []
            pnls = p.get_daily_pnl()
            for date in _daterange(p.start_datetime.date(), p.end_datetime.date()):
                capital_allocation = strategy.capitals_serie
                if isinstance(strategy.capitals_serie, dict):
                    capital_allocation = strategy.capitals_serie[date] if date in strategy.capitals_serie else 0.
                pnl = pnls[date]
                roi = 100.0 * pnl / capital_allocation if capital_allocation else 0.0
                daily_rois.append(roi)

            weekly_roi_back.append(sum(daily_rois) / len(daily_rois))

        labels = [o.start_datetime.strftime('%Y-%m-%d') for o in weekly_periods_back]
        if len(weekly_roi_prod) or len(weekly_roi_back):
            plt.plot(weekly_roi_prod,
                     'b-',
                     label='{} %ROI'.format(comparison_input.name1),
                     alpha=0.5)
            plt.plot(weekly_roi_back,
                     'y-',
                     label='{} %ROI'.format(comparison_input.name2),
                     alpha=0.5)
            plt.xticks(range(len(labels)), labels, rotation=17)
            plt.legend()
            plt.title('Weekly ROI {}'.format(strategy.get_short_name()))
            file_path = '%s/%s' % (tmp_dir, "weekly_ROI_{}.png".format(strategy.get_short_name()))
            plt.savefig(file_path)
            attachments.append(file_path)
        plt.close()

        # Weekly ROI ZScore
        roi_df = pd.DataFrame({
            '{}_ROI'.format(comparison_input.name1): weekly_roi_prod,
            '{}_ROI'.format(comparison_input.name2): weekly_roi_back,
        })

        zscores = \
            (roi_df - roi_df.rolling(window=len(roi_df), min_periods=1).mean()) / \
            roi_df.rolling(window=len(roi_df), min_periods=1).std()
        zscores.fillna(0, inplace=True)  # replaces Nan with zeroes

        labels = [o.start_datetime.strftime('%Y-%m-%d') for o in weekly_periods_back]
        if len(weekly_roi_prod) or len(weekly_roi_back):
            plt.plot(zscores['{}_ROI'.format(comparison_input.name1)],
                     'b-',
                     label='{} ZScore'.format(comparison_input.name1),
                     alpha=0.5)
            plt.plot(zscores['{}_ROI'.format(comparison_input.name2)],
                     'y-',
                     label='{} ZScore'.format(comparison_input.name2),
                     alpha=0.5)
            plt.xticks(range(len(labels)), labels, rotation=17)
            plt.legend()
            plt.title('Weekly ROI zscore {}'.format(strategy.get_short_name()))
            file_path = '%s/%s' % (tmp_dir, "weekly_ROI_zscore_{}.png".format(strategy.get_short_name()))
            plt.savefig(file_path)
            attachments.append(file_path)
        plt.close()

    message += tabulate(lines, headers=headers, stralign='right')

    res = TestResult(name="ROI {}/{}".format(comparison_input.name1, comparison_input.name2),
                     good_report_message=message,
                     attachments=attachments,
                     )
    res.success = True
    return res


def _get_extra_stats(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """
    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    headers = ["Matched vol", "Matched win", "Matched lose",
               "Tot PNL", "Win", "Lose", "Win over Matched win", "Lose over Matched Lose"]
    lines = []
    message = ""
    attachments = []
    for strategy in comparison_input.strategies:
        prod_matched_win = sum(
            [o['size_matched'] for o in strategy.orders1 if o['status'] == OrderStatus.SETTLED and _get_order_pnl(o) > 0])
        back_matched_win = sum(
            [o['size_matched'] for o in strategy.orders2 if o['status'] == OrderStatus.SETTLED and _get_order_pnl(o) > 0])
        prod_matched_lose = sum(
            [o['size_matched'] for o in strategy.orders1 if o['status'] == OrderStatus.SETTLED and _get_order_pnl(o) < 0])
        back_matched_lose = sum(
            [o['size_matched'] for o in strategy.orders2 if o['status'] == OrderStatus.SETTLED and _get_order_pnl(o) < 0])

        pnl_prod = sum([_get_order_pnl(o) for o in strategy.orders1])
        pnl_back = sum([_get_order_pnl(o) for o in strategy.orders2])
        win_prod = sum([_get_order_pnl(o) for o in strategy.orders1 if _get_order_pnl(o) > 0])
        win_back = sum([_get_order_pnl(o) for o in strategy.orders2 if _get_order_pnl(o) > 0])
        lose_prod = sum([_get_order_pnl(o) for o in strategy.orders1 if _get_order_pnl(o) < 0])
        lose_back = sum([_get_order_pnl(o) for o in strategy.orders2 if _get_order_pnl(o) < 0])

        line = [strategy.get_short_name(),
                "{:8.0f}/{:>8.0f}".format(  # Matched vol
                    sum([o['size_matched'] for o in strategy.orders1 if o['status'] == OrderStatus.SETTLED]),
                    sum([o['size_matched'] for o in strategy.orders2 if o['status'] == OrderStatus.SETTLED])),
                "{:8.0f}/{:>8.0f}".format(prod_matched_win, back_matched_win),  # Matched win
                "{:8.0f}/{:>8.0f}".format(prod_matched_lose, back_matched_lose),  # Matched lose
                "{:6.0f}/{:>6.0f}".format(pnl_prod, pnl_back),  # Tot PNL
                "{:6.0f}/{:>6.0f}".format(win_prod, win_back),  # Win
                "{:6.0f}/{:>6.0f}".format(lose_prod, lose_back),  # Lose
                "{:2.3f}/{:>2.3f}".format(0.0 if prod_matched_win == 0 else float(win_prod / prod_matched_win),
                                          0.0 if back_matched_win == 0 else float(win_back / back_matched_win)),
                # Win over Matched Win
                "{:2.3f}/{:>2.3f}".format(0.0 if prod_matched_lose == 0 else float(lose_prod / prod_matched_lose),
                                          0.0 if back_matched_lose == 0 else float(lose_back / back_matched_lose)),
                # Lose over Matched Lose
                ]
        lines.append(line)

    message += tabulate(lines, headers=headers, stralign='right')

    res = TestResult(name="Vol stats {}/{}".format(comparison_input.name1, comparison_input.name2),
                     good_report_message=message,
                     attachments=attachments,
                     )
    res.success = True
    return res


def _get_daily_roi(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """
    Create daily cumulative ROI chart

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = ""
    attachments = []
    for strategy in comparison_input.strategies:
        daily_periods_prod = strategy.daily_periods_1
        daily_periods_back = strategy.daily_periods_2

        daily_roi_prod = []
        for p in daily_periods_prod:
            daily_rois = []
            pnls = p.get_daily_pnl()
            for date in _daterange(p.start_datetime.date(), p.end_datetime.date()):
                capital_allocation = strategy.capitals_serie
                if isinstance(strategy.capitals_serie, dict):
                    capital_allocation = strategy.capitals_serie[date] if date in strategy.capitals_serie else 0.0
                pnl = pnls[date]
                roi = 100.0 * pnl / capital_allocation if capital_allocation else 0.0
                daily_rois.append(roi)

            daily_roi_prod.append(sum(daily_rois) / len(daily_rois))

        daily_roi_back = []
        for p in daily_periods_back:
            daily_rois = []
            pnls = p.get_daily_pnl()
            for date in _daterange(p.start_datetime.date(), p.end_datetime.date()):
                capital_allocation = strategy.capitals_serie
                if isinstance(strategy.capitals_serie, dict):
                    capital_allocation = strategy.capitals_serie[date] if date in strategy.capitals_serie else 0.0
                pnl = pnls[date]
                roi = 100.0 * pnl / capital_allocation if capital_allocation else 0.0
                daily_rois.append(roi)

            daily_roi_back.append(sum(daily_rois) / len(daily_rois))

        cumulative_back_roi = np.cumsum(daily_roi_back)
        cumulative_prod_roi = np.cumsum(daily_roi_prod)

        labels = [''] * len(daily_periods_prod)
        for i in range(0, len(daily_periods_prod), 7):
            labels[i] = daily_periods_prod[i].start_datetime.strftime('%Y-%m-%d')
        if len(cumulative_prod_roi) or len(cumulative_back_roi):
            plt.plot(cumulative_prod_roi,
                     'b-',
                     label='{} ROI'.format(comparison_input.name1),
                     alpha=0.5)
            plt.plot(cumulative_back_roi,
                     'y-',
                     label='{} ROI'.format(comparison_input.name2),
                     alpha=0.5)
            plt.xticks(range(len(labels)), labels, rotation=17)
            plt.legend()
            plt.title('Daily cumulative ROI {}'.format(strategy.get_short_name()))
            file_path = '%s/%s' % (tmp_dir, "daily_cumulative_ROI_{}.png".format(strategy.get_short_name()))
            plt.savefig(file_path)
            attachments.append(file_path)
        plt.close()

    res = TestResult(name="Daily ROI {}/{}".format(comparison_input.name1, comparison_input.name2),
                     good_report_message=message,
                     attachments=attachments,
                     )
    res.success = True
    return res


def _print_capital(comparison_input, start_datetime, end_datetime, tmp_dir):
    """
    Create daily cumulative ROI chart

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = "See line graph"
    attachments = []
    for strategy in comparison_input.strategies:

        dates = sorted([d for d in _daterange(start_datetime.date(), end_datetime.date())])
        labels = [''] * len(dates)
        for i in range(0, len(dates), 7):
            labels[i] = dates[i].strftime('%Y-%m-%d')

        capital_by_day = [strategy.capitals_serie[date] for date in dates] if isinstance(strategy.capitals_serie, dict) \
            else [strategy.capitals_serie for date in dates]

        if len(capital_by_day):
            plt.plot(capital_by_day,
                     'r-',
                     label='Allocated capital',
                     alpha=0.5)
            plt.xticks(range(len(labels)), labels, rotation=17)
            plt.legend()
            plt.title('Daily capital {}'.format(strategy.get_short_name()))
            file_path = '%s/%s' % (tmp_dir, "daily_capital_{}.png".format(strategy.get_short_name()))
            plt.savefig(file_path)
            attachments.append(file_path)
        plt.close()

    res = TestResult(name="Daily capital",
                     good_report_message=message,
                     attachments=attachments,
                     )
    res.success = True
    return res


def _print_capital_and_missed_volume(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """
    Create daily cumulative ROI chart

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = "See line graph"
    attachments = []
    for strategy in comparison_input.strategies:

        dates = sorted([d for d in _daterange(start_datetime.date(), end_datetime.date())])
        labels = [''] * len(dates)
        for i in range(0, len(dates), 7):
            labels[i] = dates[i].strftime('%Y-%m-%d')

        capital_by_day = [strategy.capitals_serie[date] for date in dates]

        missed_vol = {}
        missed_pnl = {}
        for k in strategy.capitals_serie.keys():
            missed_vol[k] = 0.
            missed_pnl[k] = 0.

        daily_periods_prod = split_by_time_period(strategy.instructions1, 1,
                                                  start_datetime.date(),
                                                  end_datetime.date(),
                                                  use_instructions=True,
                                                  use_cache=use_cache)

        missed_vols = []
        missed_pnl = []
        for period in daily_periods_prod:
            vol, pnl = period.get_missed_vol()
            missed_vols.append(vol)
            missed_pnl.append(pnl)

        if len(capital_by_day):
            plt.plot(capital_by_day,
                     'r-',
                     label='Allocated capital')
            plt.plot(missed_vols,
                     'y-',
                     label='missed vols prod')
            plt.plot(missed_pnl,
                     'g-',
                     label='missed pnl prod')

            plt.xticks(range(len(labels)), labels, rotation=17)
            plt.legend()
            plt.title('Daily capital and missed {}'.format(strategy.get_short_name()))
            file_path = '%s/%s' % (tmp_dir, "daily_capital_and_missed{}.png".format(strategy.get_short_name()))
            plt.savefig(file_path)
            attachments.append(file_path)
        plt.close()

    res = TestResult(name="Daily capital and missed",
                     good_report_message=message,
                     attachments=attachments,
                     )
    res.success = True
    return res


def _get_different_events(comparison_input):
    """

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = ""
    headers = ["{}_only".format(comparison_input.name1), "{}_only".format(comparison_input.name2),
               "common", "total", "diff%"]
    data = []
    for strategy in comparison_input.strategies:
        prod_events = set([parse_sticker(o['sticker'])[1][1] for o in strategy.orders1])
        backtest_events = set([parse_sticker(o['sticker'])[1][1] for o in strategy.orders2])
        common_events = prod_events & backtest_events
        union_events = prod_events | backtest_events
        n_prod_only = len(prod_events) - len(common_events)
        n_back_only = len(backtest_events) - len(common_events)

        line = list()
        line.append(strategy.get_short_name())
        line.extend([n_prod_only,
                     n_back_only,
                     len(common_events),
                     len(union_events),
                     "{:.0f}%".format(
                         ((n_prod_only + n_back_only) * 100.0 / len(union_events)) if len(union_events) else 0.0)
                     ])
        data.append(line)
    message += tabulate(data, headers=headers, stralign='right')

    res = TestResult(name="Different events".format(),
                     good_report_message=message,
                     )
    res.success = True
    return res


class PriceSlippage(object):
    def __init__(self):
        self.back_vol = 0.0
        self.back_vol_matched = 0.0
        self.back_price_times_vol = 0.0
        self.back_price_matched_times_vol = 0.0
        self.lay_vol = 0.0
        self.lay_vol_matched = 0.0
        self.lay_price_times_vol = 0.0
        self.lay_price_matched_times_vol = 0.0

        self.back_avg_price = None
        self.back_avg_matched_price = None
        self.lay_avg_price = None
        self.lay_avg_matched_price = None

        self.back_perc_match_price_vs_asked = None
        self.lay_perc_match_price_vs_asked = None

    def make_avg(self):
        self.back_avg_price = self.back_price_times_vol / self.back_vol if self.back_vol > 0.0 else 0.0
        self.back_avg_matched_price = (self.back_price_matched_times_vol / self.back_vol_matched) \
            if self.back_vol_matched > 0.0 else 0.0
        self.back_perc_match_price_vs_asked = (100.0 * self.back_avg_matched_price / self.back_avg_price) - 100 \
            if self.back_avg_price else 0.0

        self.lay_avg_price = self.lay_price_times_vol / self.lay_vol if self.lay_vol > 0.0 else 0.0
        self.lay_avg_matched_price = (self.lay_price_matched_times_vol / self.lay_vol_matched) \
            if self.lay_vol_matched > 0.0 else 0.0
        self.lay_perc_match_price_vs_asked = (100.0 * self.lay_avg_matched_price / self.lay_avg_price) - 100 \
            if self.lay_avg_price else 0.0


def _get_prices_avg_by_vol(orders):
    prices = PriceSlippage()
    for o in orders:
        if o['status'] not in [OrderStatus.SETTLED]:
            continue
        vol = _get_size(o)
        matched_vol = _get_size_matched(o)
        price = o['price']
        matched_price = o['average_price_matched']
        if o['bet_side'] == 'back':
            prices.back_vol += vol
            prices.back_vol_matched += matched_vol
            prices.back_price_times_vol += price * vol
            prices.back_price_matched_times_vol += matched_price * matched_vol
        else:
            prices.lay_vol += vol
            prices.lay_vol_matched += matched_vol
            prices.lay_price_times_vol += price * vol
            prices.lay_price_matched_times_vol += matched_price * matched_vol

    prices.make_avg()
    return prices


def _price_slippage(comparison_input):
    """

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = ""
    headers_back = ["{} price".format(comparison_input.name1),
                    "{} matched".format(comparison_input.name1),
                    "{} price".format(comparison_input.name2),
                    "{} matched".format(comparison_input.name2),
                    ]
    headers_lay = ["{} price".format(comparison_input.name1),
                   "{} matched".format(comparison_input.name1),
                   "{} price".format(comparison_input.name2),
                   "{} matched".format(comparison_input.name2),
                   ]
    data = []
    for strategy in comparison_input.strategies:
        prod_prices = _get_prices_avg_by_vol(strategy.orders1)
        back_prices = _get_prices_avg_by_vol(strategy.orders2)
        line = list()
        line.append(strategy.get_short_name())
        line.extend(["{:.2f}".format(prod_prices.back_avg_price),
                     "{:.2f}".format(prod_prices.back_avg_matched_price),
                     "{:.2f}".format(back_prices.back_avg_price),
                     "{:.2f}".format(back_prices.back_avg_matched_price),
                     ])
        data.append(line)
    message += "Back prices:\n"
    message += tabulate(data, headers=headers_back)

    data = []
    for strategy in comparison_input.strategies:
        prod_prices = _get_prices_avg_by_vol(strategy.orders1)
        back_prices = _get_prices_avg_by_vol(strategy.orders2)
        line = list()
        line.append(strategy.get_short_name())
        line.extend(["{:.2f}".format(prod_prices.lay_avg_price),
                     "{:.2f}".format(prod_prices.lay_avg_matched_price),
                     "{:.2f}".format(back_prices.lay_avg_price),
                     "{:.2f}".format(back_prices.lay_avg_matched_price),
                     ])
        data.append(line)
    message += "Lay prices:\n"
    message += tabulate(data, headers=headers_lay)

    res = TestResult(name="Price slippage".format(),
                     good_report_message=message
                     )
    res.success = True
    return res


def _get_orders_by_bookmaker(orders):
    orders_by_bookmaker = dict()
    for o in orders:
        if 'provider' in o['execution_details']:
            bm = "{}_{}".format(o['execution_details']['provider'], o['execution_details']['bookmaker'])
        else:
            bm = "{}".format(o['execution_details']['bookmaker'])
        if bm not in orders_by_bookmaker:
            orders_by_bookmaker[bm] = []
        orders_by_bookmaker[bm].append(o)
    return orders_by_bookmaker


def _price_slippage_by_venue(comparison_input):
    """

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = ""

    all_bm = []
    for strategy in comparison_input.strategies:
        all_bm.extend(_get_orders_by_bookmaker(strategy.orders1).keys())
    all_bm = list(set(all_bm))

    # Prod Back
    data = []
    for strategy in comparison_input.strategies:
        line = list()
        line.append(strategy.get_short_name())
        prod_orders_by_bm = _get_orders_by_bookmaker(strategy.orders1)
        for bm in all_bm:
            if bm in prod_orders_by_bm:
                orders = prod_orders_by_bm[bm]
                prod_prices = _get_prices_avg_by_vol(orders)
                line.extend(["{:>5.2f}%".format(prod_prices.back_perc_match_price_vs_asked)])
            else:
                line.extend(["-"])
        data.append(line)
    message += "\n{} B prices matched vs asked:\n".format(comparison_input.name1)
    message += tabulate(data, headers=all_bm, stralign='right')

    # Prod Lay
    data = []
    for strategy in comparison_input.strategies:
        line = list()
        line.append(strategy.get_short_name())
        prod_orders_by_bm = _get_orders_by_bookmaker(strategy.orders1)
        for bm in all_bm:
            if bm in prod_orders_by_bm:
                orders = prod_orders_by_bm[bm]
                prod_prices = _get_prices_avg_by_vol(orders)
                line.extend(["{:<5.2f}%".format(prod_prices.lay_perc_match_price_vs_asked)])
            else:
                line.extend(["-"])
        data.append(line)
    message += "\n\n{} L prices matched vs asked:\n".format(comparison_input.name1)
    message += tabulate(data, headers=all_bm)

    # Backtest Back
    data = []
    for strategy in comparison_input.strategies:
        line = list()
        line.append(strategy.get_short_name())
        back_orders_by_bm = _get_orders_by_bookmaker(strategy.orders2)
        for bm in all_bm:
            if bm in back_orders_by_bm:
                orders = back_orders_by_bm[bm]
                back_prices = _get_prices_avg_by_vol(orders)
                line.extend(["{:>5.2f}%".format(back_prices.back_perc_match_price_vs_asked)])
            else:
                line.extend(["-"])
        data.append(line)
    message += "\n\n{} B prices matched vs asked:\n".format(comparison_input.name2)
    message += tabulate(data, headers=all_bm, stralign='right')

    # Backtest Lay
    data = []
    for strategy in comparison_input.strategies:
        line = list()
        line.append(strategy.get_short_name())
        back_orders_by_bm = _get_orders_by_bookmaker(strategy.orders2)
        for bm in all_bm:
            if bm in back_orders_by_bm:
                orders = back_orders_by_bm[bm]
                back_prices = _get_prices_avg_by_vol(orders)
                line.extend(["{:<5.2f}%".format(back_prices.lay_perc_match_price_vs_asked)])
            else:
                line.extend(["-"])
        data.append(line)
    message += "\n\n{} L prices matched vs asked:\n".format(comparison_input.name2)
    message += tabulate(data, headers=all_bm)

    res = TestResult(name="Price slippage by venue".format(),
                     good_report_message=message
                     )
    res.success = True
    return res


def _get_order_pnl(order):
    return StrategyRunStatsHelper.get_order_pnl(order)


def _get_pnl_per_event(comparison_input):
    """

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = ""

    all_bm1 = []
    for strategy in comparison_input.strategies:
        all_bm1.extend(_get_orders_by_bookmaker(strategy.orders1).keys())
    all_bm1 = list(set(all_bm1))

    # Prod Back
    data = []
    for strategy in comparison_input.strategies:

        settled1 = [o for o in strategy.orders1 if o['status'] == OrderStatus.SETTLED]
        settled2 = [o for o in strategy.orders2 if o['status'] == OrderStatus.SETTLED]
        events_pnl1 = defaultdict(int)
        events_pnl2 = defaultdict(int)
        events_matched1 = defaultdict(int)
        events_matched2 = defaultdict(int)
        for o in settled1:
            _, (_, event_id), _, params, _ = parse_sticker(o['sticker'])
            events_pnl1[event_id] += _get_order_pnl(o)
            events_matched1[event_id] += o['size_matched'] if o['status'] == OrderStatus.SETTLED else 0
        for o in settled2:
            _, (_, event_id), _, params, _ = parse_sticker(o['sticker'])
            events_pnl2[event_id] += _get_order_pnl(o)
            events_matched2[event_id] += o['size_matched'] if o['status'] == OrderStatus.SETTLED else 0
        all_events = set()
        all_events.update(set(events_pnl1.keys()))
        all_events.update(set(events_pnl2.keys()))

        for event_id in all_events:
            line = list()
            line.append("{}".format(event_id))
            line.append("{:>5.2f}".format(events_pnl1[event_id]))
            line.append("{:>5.2f}".format(events_pnl2[event_id]))
            line.append("{:>5.2f}".format(abs(events_pnl1[event_id] - events_pnl2[event_id])))
            line.append("{:>5.2f}".format(events_matched1[event_id]))
            line.append("{:>5.2f}".format(events_matched2[event_id]))
            line.append("{:>5.2f}".format(abs(events_matched1[event_id] - events_matched2[event_id])))
            data.append(line)

        message += "\n\n PNL per event:  {}\n".format(strategy.get_short_name())
        message += tabulate(data, headers=["event", comparison_input.name1, comparison_input.name2, "diff",
                                           "Matched1", "Matched2", "Diff"], stralign='right')

    res = TestResult(name="PNL per event".format(),
                     good_report_message=message
                     )
    res.success = True
    return res


def _get_pnl_zscore(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = "See the line graph"
    attachments = []
    for strategy in comparison_input.strategies:
        periods_prod = split_by_time_period(strategy.orders1, 7,
                                            start_datetime.date(),
                                            end_datetime.date(),
                                            use_cache=use_cache)

        periods_back = split_by_time_period(strategy.orders2, 7,
                                            start_datetime,
                                            end_datetime,
                                            use_cache=use_cache)

        back_pnls = pd.Series([p.get_pnl() for p in periods_back],
                              index=[p.start_datetime.date() for p in periods_back])
        prod_pnls = pd.Series([p.get_pnl() for p in periods_prod],
                              index=[p.start_datetime.date() for p in periods_prod])
        pnl_df = pd.DataFrame({
            '{}_pnl'.format(comparison_input.name1): prod_pnls,
            '{}_pnl'.format(comparison_input.name2): back_pnls,
        })

        zscores = \
            (pnl_df - pnl_df.rolling(window=len(pnl_df), min_periods=1).mean()) / \
            pnl_df.rolling(window=len(pnl_df), min_periods=1).std()
        zscores.fillna(0, inplace=True)  # replaces Nan with zeroes

        labels = [o.start_datetime.strftime('%Y-%m-%d') for o in periods_back]
        plt.plot(zscores['{}_pnl'.format(comparison_input.name1)],
                 'b-',
                 label='{} ZScore'.format(comparison_input.name1),
                 alpha=0.5)
        plt.plot(zscores['{}_pnl'.format(comparison_input.name2)],
                 'y-',
                 label='{} ZScore'.format(comparison_input.name2),
                 alpha=0.5)
        plt.xticks(range(len(labels)), labels, rotation=17)
        plt.legend()
        plt.title('Weekly pnl zscore {}'.format(strategy.get_short_name()))
        file_path = '%s/%s' % (tmp_dir, "weekly_zscore_{}.png".format(strategy.get_short_name()))
        plt.savefig(file_path)
        attachments.append(file_path)
        plt.close()

    res = TestResult(name="ZScore by week",
                     good_report_message=message,
                     attachments=attachments)
    res.success = True
    return res


def _get_weekly_cumulative_pnl(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = "See the line graph"
    attachments = []
    for strategy in comparison_input.strategies:
        periods_prod = split_by_time_period(strategy.orders1, 7,
                                            start_datetime.date(),
                                            end_datetime.date(),
                                            use_cache=use_cache)

        periods_back = split_by_time_period(strategy.orders2, 7,
                                            start_datetime,
                                            end_datetime,
                                            use_cache=use_cache)

        back_pnls = [p.get_pnl() for p in periods_back]
        prod_pnls = [p.get_pnl() for p in periods_prod]
        cumulative_back_pnls = np.cumsum(back_pnls)
        cumulative_prod_pnls = np.cumsum(prod_pnls)

        labels = [o.start_datetime.strftime('%Y-%m-%d') for o in periods_back]
        plt.plot(cumulative_prod_pnls,
                 'b-',
                 label='{} pnl'.format(comparison_input.name1),
                 alpha=0.5)
        plt.plot(cumulative_back_pnls,
                 'y-',
                 label='{} pnl'.format(comparison_input.name2),
                 alpha=0.5)
        plt.xticks(range(len(labels)), labels, rotation=17)
        plt.legend()
        plt.title('Weekly cumulative PNL {}'.format(strategy.get_short_name()))
        file_path = '%s/%s' % (tmp_dir, "weekly_cumulative_pnl_{}.png".format(strategy.get_short_name()))
        plt.savefig(file_path)
        attachments.append(file_path)
        plt.close()

    res = TestResult(name="Weekly cumulative pnl",
                     good_report_message=message,
                     attachments=attachments)
    res.success = True
    return res


def _get_daily_cumulative_pnl(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = "See the line graph"
    attachments = []
    for strategy in comparison_input.strategies:
        periods_prod = strategy.daily_periods_1
        periods_back = strategy.daily_periods_2

        back_pnls = [p.get_pnl() for p in periods_back]
        prod_pnls = [p.get_pnl() for p in periods_prod]
        cumulative_back_pnls = np.cumsum(back_pnls)
        cumulative_prod_pnls = np.cumsum(prod_pnls)

        labels = [''] * len(periods_back)
        for i in range(0, len(periods_back), 7):
            labels[i] = periods_back[i].start_datetime.strftime('%Y-%m-%d')
        plt.plot(cumulative_prod_pnls,
                 'b-',
                 label='{} pnl'.format(comparison_input.name1),
                 alpha=0.5)
        plt.plot(cumulative_back_pnls,
                 'y-',
                 label='{} pnl'.format(comparison_input.name2),
                 alpha=0.5)
        plt.xticks(range(len(labels)), labels, rotation=17)
        plt.legend()
        plt.title('Daily cumulative PNL {}'.format(strategy.get_short_name()))
        file_path = '%s/%s' % (tmp_dir, "daily_cumulative_pnl_{}.png".format(strategy.get_short_name()))
        plt.savefig(file_path)
        attachments.append(file_path)
        plt.close()

    res = TestResult(name="Daily cumulative pnl",
                     good_report_message=message,
                     attachments=attachments)
    res.success = True
    return res


def _get_daily_pnl(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = "See the line graph"
    attachments = []
    for strategy in comparison_input.strategies:
        periods_prod = strategy.daily_periods_1
        periods_back = strategy.daily_periods_2

        back_pnls = [p.get_pnl() for p in periods_back]
        prod_pnls = [p.get_pnl() for p in periods_prod]

        labels = [''] * len(periods_back)
        for i in range(0, len(periods_back), 7):
            labels[i] = periods_back[i].start_datetime.strftime('%Y-%m-%d')
        plt.plot(prod_pnls,
                 'b-',
                 label='{} pnl'.format(comparison_input.name1),
                 alpha=0.5)
        plt.plot(back_pnls,
                 'y-',
                 label='{} pnl'.format(comparison_input.name2),
                 alpha=0.5)
        plt.xticks(range(len(labels)), labels, rotation=17)
        plt.legend()
        plt.title('Daily PNL {}'.format(strategy.get_short_name()))
        file_path = '%s/%s' % (tmp_dir, "Daily_pnl_{}.png".format(strategy.get_short_name()))
        plt.savefig(file_path)
        attachments.append(file_path)
        plt.close()

    res = TestResult(name="Daily pnl",
                     good_report_message=message,
                     attachments=attachments)
    res.success = True
    return res


def _get_daily_cumulative_vol(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache):
    """

    :param comparison_input: type [ComparisonInput]
    :return: TestResult
    """

    message = "See the line graph"
    attachments = []
    for strategy in comparison_input.strategies:
        periods_prod = strategy.daily_periods_1
        periods_back = strategy.daily_periods_2

        back_vols = [p.get_matched_vol() for p in periods_back]
        prod_vols = [p.get_matched_vol() for p in periods_prod]
        cumulative_back_vols = np.cumsum(back_vols)
        cumulative_prod_vols = np.cumsum(prod_vols)

        labels = [''] * len(periods_back)
        for i in range(0, len(periods_back), 7):
            labels[i] = periods_back[i].start_datetime.strftime('%Y-%m-%d')
        plt.plot(cumulative_prod_vols,
                 'b-',
                 label='{} pnl'.format(comparison_input.name1),
                 alpha=0.5)
        plt.plot(cumulative_back_vols,
                 'y-',
                 label='{} pnl'.format(comparison_input.name2),
                 alpha=0.5)
        plt.xticks(range(len(labels)), labels, rotation=17)
        plt.legend()
        plt.title('Daily cumulative matched vol {}'.format(strategy.get_short_name()))
        file_path = '%s/%s' % (tmp_dir, "daily_cumulative_vol_{}.png".format(strategy.get_short_name()))
        plt.savefig(file_path)
        attachments.append(file_path)
        plt.close()

    res = TestResult(name="Daily cumulative vol",
                     good_report_message=message,
                     attachments=attachments)
    res.success = True
    return res


def _print_orders(comparison_input, start_datetime, end_datetime):
    message = ""
    for strategy in comparison_input.strategies:
        message += "{}\n".format(strategy.get_short_name())
        message += _get_orders_debug_print(strategy.orders1, strategy.orders2, {},
                                           strategy.strategy_name, strategy.strategy_desc,
                                           strategy.strategy_code, strategy.trading_user_id,
                                           start_datetime, end_datetime,
                                           name1=comparison_input.name1, name2=comparison_input.name2
                                           )

    res = TestResult(name="Orders",
                     good_report_message=message)
    res.success = True
    return res


class Report(object):
    def __init__(self, results, start, end):
        self.results = results
        self.simulated_start = start
        self.simulated_end = end
        self.messages = []

    def send_email(self, notifiees):
        subject = "Weekly report {}_{} {} days".format(self.simulated_start.strftime('%Y-%m-%d'),
                                                       self.simulated_end.strftime('%Y-%m-%d'),
                                                       (self.simulated_end - self.simulated_start).days + 1)
        body = self._make_email_body()
        content = body
        attachments = []
        for r in self.results:
            attachments.extend(r.attachments)
        self.send_report_emails(subject, content, notifiees, attachments)

    def _make_email_body(self):
        body = ""
        for r in self.results:
            if "" != r.good_report_message:
                section = "{}\n{}:\n{}\n".format("-------------", r.test_name, r.good_report_message)
                body += "{}\n".format(section)

        if self.messages:
            body += "\nExtra info:\n"
            for m in self.messages:
                body += "{}\n".format(m)
        return body

    def send_report_emails(self, subject, message, notifiees, attachments):
        start_date_str = self.simulated_start.strftime('%Y-%m-%d')
        end_date_str = self.simulated_end.strftime('%Y-%m-%d')
        num_days = (self.simulated_end - self.simulated_start).days + 1
        template = Template("""
        Report from {{start_date}} to {{end_date}} - {{num_days}} days\n

        <pre>
        \n{{ message }}\n
        </pre>
                """)
        html_ = template.render(start_date=start_date_str, end_date=end_date_str, num_days=num_days, message=message)
        _send_email_retry('', html_, subject, notifiees, attachments=attachments)

    def add_extra_messages(self, messages):
        """
        Ad extra messages to be sent in the email
        :param messages:
        :return:
        """
        self.messages.extend(messages)


class StrategyInput(object):
    def __init__(self, strategy_name, strategy_desc, trading_user_id, strategy_code):
        self.strategy_name = strategy_name
        self.strategy_desc = strategy_desc
        self.trading_user_id = trading_user_id
        self.strategy_code = strategy_code
        self.orders1 = list()
        self.orders2 = list()
        self.daily_periods_1 = list()
        self.daily_periods_2 = list()
        self.capitals_serie = None  # Not used in running but only for stats. Can be a serie or fixed amount
        self.instructions1 = list()
        self.instructions2 = list()

    def __str__(self):
        return "{} {} {} {}".format(self.strategy_name, self.strategy_desc, self.trading_user_id, self.strategy_code)

    def get_short_name(self):
        message = "{}".format(self.strategy_name)
        if self.strategy_desc is not None:
            message += " {}".format(self.strategy_desc)
        if self.strategy_code is not None:
            message += " {}".format(self.strategy_code)
        if self.trading_user_id is not None:
            if self.trading_user_id in TRADING_USER_MAP:
                message += " {}".format(TRADING_USER_MAP[self.trading_user_id])
            else:
                message += " {}".format(TRADING_USER_MAP[self.trading_user_id])
        return message


class ComparisonInput(object):
    def __init__(self, strategies):
        self.strategies = strategies  # orders in different strategies can contain duplicates
        self.all_orders1 = []  # must not contains duplicates
        self.all_orders2 = []  # must not contains duplicates
        self.name1 = ""
        self.name2 = ""


def make_report_automatic_backtesting(start_datetime, end_datetime, strategies, cmd_line, use_cache):
    """
    Fetch production and automatic_backtest orders then run the comparison

    :param start_datetime:
    :param end_datetime:
    :param strategies: type [StrategyInput]
    :param cmd_line:
    :param use_cache: use local file caches for fixture ids
    :return:
    """
    mongo_helper = persistence.MongoStrategyHelper()
    tmp_dir = tempfile.mkdtemp(suffix="weekly_report")

    loaded = mongo_helper.get_prod_orders_between_datetimes(start_datetime - dt.timedelta(hours=12),
                                                             end_datetime + dt.timedelta(hours=12),
                                                             use_cache=use_cache)
    # loaded = [o for o in loaded if str(o['trading_user_id']) in TRADING_USER_MAP.keys()]
    all_backtest_orders = []

    for strategy in strategies:
        orders = loaded
        if strategy.strategy_name is not None:
            orders = [o for o in orders if 'strategy' in o and o['strategy'] == strategy.strategy_name]
        if strategy.strategy_desc is not None:
            orders = [o for o in orders if 'strategy_descr' in o and o['strategy_descr'] == strategy.strategy_desc]
        if strategy.trading_user_id is not None:
            orders = [o for o in orders if
                      'trading_user_id' in o and str(o['trading_user_id']) == strategy.trading_user_id]
        if strategy.strategy_code is not None:
            orders = [o for o in orders if 'strategy_code' in o and o['strategy_code'] == strategy.strategy_code]
        strategy.orders1 = orders

        instructions_backtest, orders_backtest = mongo_helper.get_backtest_result_multiple_days(
            strategy.strategy_name, strategy.strategy_desc, strategy.trading_user_id, strategy.strategy_code,
            start_datetime.date().strftime('%Y-%m-%d'),
            end_datetime.date().strftime('%Y-%m-%d'),
            'automatic', 'test_range_1')
        orders_backtest = _get_last_backtest_order_status(orders_backtest)
        strategy.orders2 = orders_backtest

        all_backtest_orders.extend(orders_backtest)

    comparison_input = ComparisonInput(strategies)
    comparison_input.name1 = 'prod'
    comparison_input.name2 = 'auto_back'
    non_duplicate_ords1 = []
    ords1_keys = set()

    for s in strategies:
        for o in s.orders1:
            if o['_id'] not in ords1_keys:
                ords1_keys.add(o['_id'])
                non_duplicate_ords1.append(o)
    comparison_input.all_orders1 = non_duplicate_ords1
    non_duplicate_ords2 = list(np.unique(np.array(all_backtest_orders)))
    comparison_input.all_orders2 = non_duplicate_ords2

    report = make_report_comparison(start_datetime, end_datetime, comparison_input, tmp_dir, use_cache)
    _ = cmd_line
    return report


def _run_for_strategy(start_datetime, end_datetime, strategies, cmd_line, repopulate, notifiees, use_fixture_cache,
                      mnemonic='report', action_delay_ms=0, use_spark=False, devpi_user=None, devpi_index=None,
                      spark_app_name=None, use_multiprocess=False, n_workers=0, extra_strategy_args=None, capital=None,
                      strategy_factory=StrategyFactory, framework_providers=FrameworkHistoricalProviders):
    """
    :param action_delay_ms: add this delay to action data
    :return: list of error messages
    """
    if extra_strategy_args is None:
        extra_strategy_args = {}
    range_to_run = (start_datetime, end_datetime, {}, 'report_range_1')

    just_repopulated_strategies = list()
    config_id_str = None
    errors = []

    for strategy in strategies:
        logging.info("Runs for {} mnemonic={}".format(str(strategy), mnemonic))
        logging.info("Runs for {}".format(str(strategy)))
        if strategy.strategy_name in ['Analyst_FTOUG', 'Analyst_FTAHG']:
            _extra_strategy_args = {'is_backtest': True}
        else:
            _extra_strategy_args = {}
        _extra_strategy_args.update(extra_strategy_args)

        if strategy.strategy_name in ['Analyst_FTOUG', 'Analyst_FTAHG', 'FFM_FTOUG', 'FFM_FTAHG']:
            _extra_strategy_args.update({'algo_type': 'SimpleAlgo'})

        if strategy.strategy_desc is not None:
            strategy_descs = [strategy.strategy_desc]
        else:
            strategy_class = strategy_factory.create_strategy_class_no_cheks(strategy.strategy_name,
                                                                            strategy.strategy_desc,
                                                                            strategy.strategy_code,
                                                                            strategy.trading_user_id,
                                                                            True, {})
            strategy_descs = strategy_class.get_valid_strategy_desc(strategy.strategy_name)
            strategy_descs = [o for o in strategy_descs if o != 'all']
        for strategy_desc in strategy_descs:
            trading_user_ids = [strategy.trading_user_id]
            if strategy.trading_user_id is None:
                trading_user_ids = ['562f5bef497aee1c22000001',  # :'Algosports',
                                    '54da2b5fd47e6bff0dade9b4']  # : 'Stratagem',

            for trading_user_id in trading_user_ids:
                strategy_codes = [strategy.strategy_code]
                if strategy.strategy_code is None:
                    strategy_class = strategy_factory.create_strategy_class_no_cheks(strategy.strategy_name,
                                                                                    strategy.strategy_desc,
                                                                                    strategy.strategy_code,
                                                                                    strategy.trading_user_id,
                                                                                    True, {})
                    strategy_codes = strategy_class.get_valid_strategy_code(strategy.strategy_name, strategy_desc)

                for strategy_code in strategy_codes:
                    extra_backtest_args = {
                        'repopulate': repopulate,
                        'use_fixture_cache': use_fixture_cache,
                        'allocated_capital': capital,  # Is not None, use constant capital for every backtest
                        'use_spark': use_spark,
                        'days_after': 1,
                        'save_n_days': 7,
                        'spark_driver_memory': '8g',
                        'spark_executor_memory': '8g',
                        'spark_app_name': "backt {} {} {} {}_{} {}".format(
                            strategy.strategy_name, strategy_desc, strategy_code,
                            start_datetime.date(), end_datetime.date(), mnemonic)
                        if spark_app_name is None else spark_app_name,
                        'devpi_user': devpi_user,
                        'devpi_index': devpi_index,
                        'action_delay_ms': action_delay_ms,
                        'use_multiprocess': use_multiprocess,
                        'n_workers': n_workers,
                        'store_signals': False,
                    }
                    try:
                        success = True
                        logging.info("Running {} {} {} {}".format(strategy.strategy_name, strategy_desc, strategy_code,
                                                                  trading_user_id))

                        extra_backtest_args_copy = deepcopy(extra_backtest_args)
                        if repopulate:
                            k = (strategy.strategy_name, strategy_desc, strategy_code, trading_user_id)
                            if k in just_repopulated_strategies:
                                # Avoid repopulating and running multiple times
                                extra_backtest_args_copy.update({'repopulate': False})
                                logging.info("Strategy {} just repopulated, not running again".format(str(k)))

                        config_id = run_backtest_main(strategy.strategy_name, strategy_desc, strategy_code,
                                                      trading_user_id,
                                                      _extra_strategy_args, extra_backtest_args_copy, config_id_str,
                                                      mnemonic, cmd_line, {}, range=range_to_run,
                                                      strategy_factory=strategy_factory,
                                                      framework_providers=framework_providers)
                        if repopulate:
                            k = (strategy.strategy_name, strategy_desc, strategy_code, trading_user_id)
                            just_repopulated_strategies.append(k)

                    except Exception as e:
                        message = "Could not run {} {} {} {}\n".format(strategy.strategy_name, strategy_desc,
                                                                       strategy_code, trading_user_id)
                        message += "From {} to {}\n".format(range_to_run[0], range_to_run[1])
                        message += "Error is: {}\n".format(e.message)
                        tb = traceback.format_exc()
                        message += "{}\n".format(tb)

                        logging.error(message)
                        logging.error("{}\n".format(tb))
                        subject = "Error while running for report"
                        _send_email_retry(message, message, subject, notifiees)
                        errors.append(message)

                    logging.info("Run {} {} {} {} done".format(strategy.strategy_name, strategy_desc, strategy_code,
                                                               trading_user_id))
    return errors


def _get_capital_timeserie(strategy_name, strategy_desc, strategy_code, trading_user_id, start_datetime, end_datetime,
                           strategy_factory=StrategyFactory):
    strategy_obj, strategy_class = strategy_factory.create_strategy(strategy_name,
                                                                   strategy_desc,
                                                                   strategy_code,
                                                                   trading_user_id,
                                                                   True, {})
    sport = strategy_obj.get_sport()
    trading_user_id = trading_user_id
    capitals_series = None
    if capitals_series is None:
        def try_get_capital_timeseries(tr_usr_id, in_sport, str_style, start_bkt, end_bkt, str_name, descr, default):
            try:
                ret = get_capital_timeseries(tr_usr_id, in_sport, str_style, start_bkt, end_bkt, str_name,
                                             strategy_descr=descr)
                return ret
            except Exception:
                return default

        # Get the historical capital time series, updated daily
        capitals_series = {descr: try_get_capital_timeseries(trading_user_id, sport,
                                                             StrategyStyle.to_str(strategy_obj.get_style()),
                                                             start_datetime,
                                                             end_datetime, strategy_name,
                                                             descr, defaultdict(lambda: 0))
                           for descr in strategy_obj.strategy_run_ids}
    return capitals_series


def _add_historical_capital_series(strategies, start_datetime, end_datetime, strategy_factory=StrategyFactory):
    for strategy in strategies:
        strategy_series = list()
        if strategy.strategy_desc is not None:
            strategy_descs = [strategy.strategy_desc]
        else:
            strategy_class = strategy_factory.create_strategy_class_no_cheks(strategy.strategy_name,
                                                                            strategy.strategy_desc,
                                                                            strategy.strategy_code,
                                                                            strategy.trading_user_id,
                                                                            True, {})
            strategy_descs = strategy_class.get_valid_strategy_desc(strategy.strategy_name)
            # Special case for Analyst
            strategy_descs = [d for d in strategy_descs if d != 'all']
        for strategy_desc in strategy_descs:
            trading_user_ids = [strategy.trading_user_id]
            if strategy.trading_user_id is None:
                trading_user_ids = ['562f5bef497aee1c22000001',  # :'Algosports',
                                    '54da2b5fd47e6bff0dade9b4']  # : 'Stratagem',

            for trading_user_id in trading_user_ids:
                strategy_codes = [strategy.strategy_code]
                if strategy.strategy_code is None:
                    strategy_class = strategy_factory.create_strategy_class_no_cheks(strategy.strategy_name,
                                                                                    strategy_desc,
                                                                                    strategy.strategy_code,
                                                                                    strategy.trading_user_id,
                                                                                    True, {})
                    strategy_codes = strategy_class.get_valid_strategy_code(strategy.strategy_name, strategy_desc)
                # Capital allocation are shared among all strategy codes
                strategy_code = strategy_codes[0]
                s = _get_capital_timeserie(strategy.strategy_name, strategy_desc, strategy_code, trading_user_id,
                                           start_datetime, end_datetime, strategy_factory=strategy_factory)
                strategy_series.append(s)

        # sum all allocations
        all_key_dates = []
        for d in strategy_series:
            for strat, dates in d.iteritems():
                all_key_dates.extend(dates.keys())
        all_key_dates = sorted(list(set(all_key_dates)))
        new_serie = dict()
        for k in all_key_dates:
            if k not in new_serie:
                new_serie[k] = 0.0
            for d in strategy_series:
                for strat, dates in d.iteritems():
                    if k in dates:
                        new_serie[k] += dates[k]

        for date in _daterange(start_datetime.date(), end_datetime.date()):
            if date not in new_serie:
                new_serie[date] = 0.0

        strategy.capitals_serie = new_serie


def _load_prod_orders(start_datetime, end_datetime, use_cache):
    _mongo_helper = persistence.MongoStrategyHelper()
    orders = _mongo_helper.get_prod_orders_between_datetimes(start_datetime - dt.timedelta(hours=12),
                                                             end_datetime + dt.timedelta(hours=12),
                                                             use_cache=use_cache)
    orders = [o for o in orders if 'strategy' in o and 'sticker' in o and 'ANTONIS' not in o['strategy']]
    orders = [o for o in orders if str(o['trading_user_id']) in ['54da2b5fd47e6bff0dade9b4',
                                                                 '562f5bef497aee1c22000001']]

    sports = list(set(sticker_parts_from_sticker(i['sticker']).sport for i in orders))
    fixtures = persistence.fetch_fixtures_ids(start_datetime, end_datetime,
                                              sports,
                                              use_cache=use_cache)
    fixture_ids = set(f_id for _, f_id in fixtures)
    orders = [o for o in orders if sticker_parts_from_sticker(o['sticker']).scope[1] in fixture_ids]
    # Apply strategy_code where needed, fetching it from the instruction
    instr_ids_to_query = []
    orders_to_reparse = []
    for o in orders:
        if o['strategy'] in ['tennis_sip_ATP', 'tennis_sip_WTA', 'tennis_sip_v2_ATP', 'tennis_sip_v2_WTA',
                             'tennis_lip_ATP', 'tennis_lip_WTA', 'bball_pbp', 'tennis_tot_games_ATP',
                             'tennis_sip_template_ATP', 'tennis_sip_template_WTA',
                             'tennis_deadball_ensemble']:
            if 'instruction_id' in o:
                instr_ids_to_query.append(o['instruction_id'])
            orders_to_reparse.append(o)
        else:
            o.update({'strategy_code': None})
    instr_ids_to_query = list(set(instr_ids_to_query))
    logging.info("Loading {} instructions...".format(len(instr_ids_to_query)))
    instructions_map = _mongo_helper.get_prod_instructions_by_ids(instr_ids_to_query)
    for o in orders_to_reparse:
        strategy_code = None
        if 'instruction_id' in o:
            if str(o['instruction_id']) in instructions_map:
                instruction = instructions_map[str(o['instruction_id'])]
                strategy_code = get_strategy_code_from_instruction(instruction)
            else:
                logging.error("Instruction {} not in map for order {}".format(str(o['instruction_id']), o))
        o.update({'strategy_code': strategy_code})
    return orders


def _load_prod_instructions(start_datetime, end_datetime):
    _mongo_helper = persistence.MongoStrategyHelper()
    instr1 = _mongo_helper.get_prod_instructions_between_datetimes(start_datetime - dt.timedelta(hours=12),
                                                                   end_datetime + dt.timedelta(hours=12),
                                                                   trading_user_id='562f5bef497aee1c22000001')
    instr2 = _mongo_helper.get_prod_instructions_between_datetimes(start_datetime - dt.timedelta(hours=12),
                                                                   end_datetime + dt.timedelta(hours=12),
                                                                   trading_user_id='54da2b5fd47e6bff0dade9b4')

    instr = [i for i in instr1 + instr2 if 'strategy_descr' in i]
    return instr


def _load_backtest_orders(strategy, start_datetime, end_datetime, mnemonic, extra_params=None,
                          range_name='report_range_1', config_id=None, strategy_factory=StrategyFactory):
    strategy_class = strategy_factory.create_strategy_class_no_cheks(strategy.strategy_name, strategy.strategy_desc,
                                                                    strategy.strategy_code,
                                                                    strategy.trading_user_id,
                                                                    True, {})

    # just 1 config per strategy_code
    mongo_helper = persistence.MongoStrategyHelper()
    instr = []
    orders = []

    if strategy.strategy_desc is not None:
        strategy_descs = [strategy.strategy_desc]
    else:
        strategy_descs = strategy_class.get_valid_strategy_desc(strategy.strategy_name)
        # Special case for Analyst
        strategy_descs = [d for d in strategy_descs if d != 'all']

    for strategy_desc in strategy_descs:
        if strategy.trading_user_id is not None:
            trading_user_ids = [strategy.trading_user_id]
        else:
            trading_user_ids = ['562f5bef497aee1c22000001',  # :'Algosports',
                                '54da2b5fd47e6bff0dade9b4']  # : 'Stratagem',

        for trading_user_id in trading_user_ids:
            if strategy.strategy_code is not None:
                strategy_codes = [strategy.strategy_code]
            else:
                strategy_codes = strategy_class.get_valid_strategy_code(strategy.strategy_name, strategy_desc)

            for strategy_code in strategy_codes:
                config = strategy_class.get_default_configuration(strategy.strategy_name, strategy_desc,
                                                                  strategy_code, trading_user_id)
                if extra_params is not None:
                    config.update(extra_params)

                if config_id is None:
                    config_id = mongo_helper.ensure_configurations(strategy.strategy_name, strategy_desc, strategy_code,
                                                                   config)

                instructions_fetched, orders_fetched = mongo_helper.get_backtest_result_multiple_days(
                    strategy.strategy_name, strategy_desc, trading_user_id, strategy_code,
                    start_datetime.date().strftime('%Y-%m-%d'),
                    end_datetime.date().strftime('%Y-%m-%d'),
                    mnemonic=mnemonic, range_name=range_name, config_id=config_id)
                instr.extend(instructions_fetched)
                orders.extend(orders_fetched)

    return instr, orders


def _get_last_backtest_order_status(in_orders):
    """
    Only keep the last orderState for every order.
    Assume that 'strategy_desc' has been added to the orders
    :param in_orders: order states as inputs
    :return:
    """
    # bet_id is unreliable, since it depends from which day we had run from and to

    kept_keys = set()
    kept_orders = list()
    for o in in_orders:
        if o['status'] in [OrderStatus.SETTLED, OrderStatus.CANCELLED]:
            key = (o['strategy_desc'], o['sticker'], o['placed_time'], o['size'])
            kept_keys.add(key)
            kept_orders.append(o)
    for o in in_orders:
        if o['status'] not in [OrderStatus.SETTLED, OrderStatus.CANCELLED]:
            key = (o['strategy_desc'], o['sticker'], o['placed_time'], o['size'])
            if key not in kept_keys:
                kept_orders.append(o)
    return kept_orders


def make_report_new_run(start_datetime, end_datetime, strategies, cmd_line, repopulate, notifiees, use_cache,
                        devpi_user=None, devpi_index=None, strategy_factory=StrategyFactory,
                        framework_providers=FrameworkHistoricalProviders):
    """
    Fetch production and backtesting orders then run the comparison

    :param strategies: type [StrategyInput]
    :param start_datetime:
    :param end_datetime:
    :param cmd_line: str, how we called this script
    :param repopulate: if true replace the database runs
    :param notifiees: list of email addresses to be notified for bug only
    :param use_cache: if True use local file cache for fixture_ids and mongo orders
    :return:
    """

    run_errors = _run_for_strategy(start_datetime, end_datetime, strategies, cmd_line, repopulate, notifiees, use_cache,
                                   devpi_user=devpi_user, devpi_index=devpi_index, strategy_factory=strategy_factory,
                                   framework_providers=framework_providers)

    _add_historical_capital_series(strategies, start_datetime, end_datetime, strategy_factory=strategy_factory)

    tmp_dir = tempfile.mkdtemp(suffix="weekly_report")

    loaded_prod_orders = _load_prod_orders(start_datetime, end_datetime, use_cache)
    loaded_prod_instructions = []  # _load_prod_instructions(start_datetime, end_datetime)

    all_backtest_orders = []

    for strategy in strategies:
        strategy_class = strategy_factory.create_strategy_class_no_cheks(strategy.strategy_name, strategy.strategy_desc,
                                                                        strategy.strategy_code,
                                                                        strategy.trading_user_id,
                                                                        True, {})

        instr = loaded_prod_instructions
        orders = loaded_prod_orders
        if strategy.strategy_name is not None:
            orders = [o for o in orders if 'strategy' in o and o['strategy'] == strategy.strategy_name]
            instr = [o for o in instr if 'strategy' in o and o['strategy'] == strategy.strategy_name]
        if strategy.strategy_desc is not None:
            orders = [o for o in orders if 'strategy_descr' in o and o['strategy_descr'] == strategy.strategy_desc]
            instr = [o for o in instr if 'strategy_descr' in o and o['strategy_descr'] == strategy.strategy_desc]
        if strategy.trading_user_id is not None:
            orders = [o for o in orders if
                      'trading_user_id' in o and str(o['trading_user_id']) == strategy.trading_user_id]
            instr = [o for o in instr if
                     'trading_user_id' in o and str(o['trading_user_id']) == strategy.trading_user_id]
        if strategy.strategy_code is not None:
            orders = [o for o in orders if 'strategy_code' in o and o['strategy_code'] == strategy.strategy_code]
            instr = [o for o in instr if 'strategy_code' in o and o['strategy_code'] == strategy.strategy_code]

        # Remove strategy_descs no longer used
        if strategy.strategy_desc is None:
            orders = [o for o in orders if 'strategy_descr' in o and
                      o['strategy_descr'] in strategy_class.get_valid_strategy_desc(strategy.strategy_name)]
            instr = [o for o in instr if 'strategy_descr' in o and
                     o['strategy_descr'] in strategy_class.get_valid_strategy_desc(strategy.strategy_name)]

        strategy.orders1 = orders
        strategy.instructions1 = instr
        instructions_backtest, orders_backtest = _load_backtest_orders(strategy, start_datetime, end_datetime, 'report',
                                                                       strategy_factory=strategy_factory)

        # Only keep last order status
        orders_backtest = _get_last_backtest_order_status(orders_backtest)
        # 'all' for Analyst is unsupported here for report
        orders_backtest = [o for o in orders_backtest if o['strategy_desc'] != 'all']
        strategy.orders2 = orders_backtest

        all_backtest_orders.extend(orders_backtest)

    comparison_input = ComparisonInput(strategies)
    comparison_input.name1 = 'prod'
    comparison_input.name2 = 'back'
    non_duplicate_ords1 = []
    ords1_keys = set()

    for s in strategies:
        for o in s.orders1:
            if o['_id'] not in ords1_keys:
                ords1_keys.add(o['_id'])
                non_duplicate_ords1.append(o)
    comparison_input.all_orders1 = non_duplicate_ords1
    non_duplicate_ords2 = list(np.unique(np.array(all_backtest_orders)))
    comparison_input.all_orders2 = non_duplicate_ords2

    report = make_report_comparison(start_datetime, end_datetime, comparison_input, tmp_dir, use_cache)
    report.add_extra_messages(run_errors)
    report.add_extra_messages(["Note: the backtesting strategy has been run with the CURRENT configuration, "
                               "not the historical one",
                               "Price slippage is computed on order's price, which for certain strategy is "
                               "worse than market price.",
                               "Football_analyst historical version has multiple competitions, while the current one "
                               "has fewer competitions."])
    return report


def make_report_new_run_action_delay(start_datetime, end_datetime, strategies, cmd_line, repopulate, notifiees,
                                     use_cache, use_spark=False, devpi_user="", devpi_index="", delay_ms=0,
                                     use_multiprocess=False, n_workers=0, strategy_factory=StrategyFactory):
    """
    Compare backtesting with and without action data delays

    :param strategies: type [StrategyInput]
    :return:
    """

    mnemonic2 = 'testing_action_delays_{}'.format(delay_ms)
    run_errors2 = _run_for_strategy(start_datetime, end_datetime, strategies, cmd_line, repopulate, notifiees,
                                    use_cache, mnemonic=mnemonic2, action_delay_ms=delay_ms, use_spark=use_spark,
                                    devpi_user=devpi_user, devpi_index=devpi_index, spark_app_name=mnemonic2,
                                    use_multiprocess=use_multiprocess, n_workers=n_workers,
                                    strategy_factory=strategy_factory)
    # run_errors2 = []

    mnemonic1 = 'testing_no_action_delays'
    run_errors1 = _run_for_strategy(start_datetime, end_datetime, strategies, cmd_line, repopulate, notifiees,
                                    use_cache, mnemonic=mnemonic1, action_delay_ms=0, use_spark=use_spark,
                                    devpi_user=devpi_user, devpi_index=devpi_index, spark_app_name=mnemonic1,
                                    use_multiprocess=use_multiprocess, n_workers=n_workers,
                                    strategy_factory=strategy_factory)

    # run_errors1 = []

    _add_historical_capital_series(strategies, start_datetime, end_datetime, strategy_factory=strategy_factory)
    tmp_dir = tempfile.mkdtemp(suffix="action_delay")

    all_no_delay_orders = []
    all_delay_orders = []

    for strategy in strategies:
        # load no delays
        instructions_no_delays, orders_no_delays = _load_backtest_orders(strategy, start_datetime, end_datetime,
                                                                         mnemonic1, strategy_factory=strategy_factory)
        orders_no_delays = _get_last_backtest_order_status(orders_no_delays)
        # 'all' for Analyst is unsupported here for report
        orders_no_delays = [o for o in orders_no_delays if o['strategy_desc'] != 'all']
        instructions_no_delays = [o for o in instructions_no_delays if o['strategy_descr'] != 'all']
        strategy.orders1 = orders_no_delays
        strategy.instructions1 = instructions_no_delays
        all_no_delay_orders.extend(orders_no_delays)

        # load delays
        instructions_delays, orders_delays = _load_backtest_orders(strategy, start_datetime, end_datetime,
                                                                   mnemonic2, strategy_factory=strategy_factory)
        orders_delays = _get_last_backtest_order_status(orders_delays)
        # 'all' for Analyst is unsupported here for report
        orders_delays = [o for o in orders_delays if o['strategy_desc'] != 'all']
        instructions_delays = [o for o in instructions_delays if o['strategy_descr'] != 'all']
        strategy.orders2 = orders_delays
        strategy.instructions2 = instructions_delays
        all_delay_orders.extend(orders_delays)

    comparison_input = ComparisonInput(strategies)
    comparison_input.name1 = 'no_action_delays'
    comparison_input.name2 = 'action_delays_{}'.format(delay_ms)

    non_duplicate_ords1 = list(np.unique(np.array(all_no_delay_orders)))
    comparison_input.all_orders1 = non_duplicate_ords1
    non_duplicate_ords2 = list(np.unique(np.array(all_delay_orders)))
    comparison_input.all_orders2 = non_duplicate_ords2

    report = make_report_comparison(start_datetime, end_datetime, comparison_input, tmp_dir, use_cache)
    report.add_extra_messages(run_errors1 + run_errors2)
    report.add_extra_messages(["Note: the backtesting strategy has been run with the CURRENT configuration, "
                               "not the historical one",
                               "Price slippage is computed on order's price, which for certain strategy is "
                               "worse than market price."])
    return report


def make_report_analyst(start_datetime, end_datetime, strategies, cmd_line, repopulate, notifiees, use_cache,
                        use_spark=False, devpi_user="", devpi_index="", use_multiprocess=False, n_workers=0,
                        strategy_factory=StrategyFactory):
    """
    Compare backtesting with different strategy args for sizing

    :param strategies: type [StrategyInput]
    :return:
    """

    mnemonic1 = 'testing_sizing_flat_0005'
    extra_strategy_args1 = {
        'type_sizing': 'flat',
        'fix_size': 0.005
    }

    run_errors1 = _run_for_strategy(start_datetime, end_datetime, strategies, cmd_line, repopulate, notifiees,
                                    use_cache, mnemonic=mnemonic1, use_spark=use_spark, devpi_user=devpi_user,
                                    devpi_index=devpi_index, spark_app_name=mnemonic1,
                                    use_multiprocess=use_multiprocess,
                                    n_workers=n_workers, extra_strategy_args=extra_strategy_args1,
                                    strategy_factory=strategy_factory)

    mnemonic2 = 'testing_sizing_rich'.format()
    extra_strategy_args2 = {
        #    'sizing': 'flat'
    }
    run_errors2 = _run_for_strategy(start_datetime, end_datetime, strategies, cmd_line, repopulate, notifiees,
                                    use_cache, mnemonic=mnemonic2, use_spark=use_spark, devpi_user=devpi_user,
                                    devpi_index=devpi_index, spark_app_name=mnemonic2,
                                    use_multiprocess=use_multiprocess,
                                    n_workers=n_workers, extra_strategy_args=extra_strategy_args2,
                                    strategy_factory=strategy_factory)

    # Alternatively, use flat capital allocation setting capital=100000
    _add_historical_capital_series(strategies, start_datetime, end_datetime, strategy_factory=strategy_factory)

    tmp_dir = tempfile.mkdtemp(suffix="analyst_sizing")

    orders1 = []
    orders2 = []

    for strategy in strategies:
        # load 1
        instructions1, orders1 = _load_backtest_orders(strategy, start_datetime, end_datetime, mnemonic1,
                                                       strategy_factory=strategy_factory)
        orders1 = _get_last_backtest_order_status(orders1)
        # 'all' for Analyst is unsupported here for report
        orders1 = [o for o in orders1 if o['strategy_desc'] != 'all']
        instructions1 = [o for o in instructions1 if o['strategy_descr'] != 'all']
        strategy.orders1 = orders1
        strategy.instructions1 = instructions1

        # load 2
        instruction2, orders2 = _load_backtest_orders(strategy, start_datetime, end_datetime, mnemonic2,
                                                      strategy_factory=strategy_factory)
        orders2 = _get_last_backtest_order_status(orders2)
        # 'all' for Analyst is unsupported here for report
        orders2 = [o for o in orders2 if o['strategy_desc'] != 'all']
        instruction2 = [o for o in instruction2 if o['strategy_descr'] != 'all']
        strategy.orders2 = orders2
        strategy.instructions2 = instruction2

    comparison_input = ComparisonInput(strategies)
    comparison_input.name1 = mnemonic1
    comparison_input.name2 = mnemonic2

    non_duplicate_ords1 = list(np.unique(np.array(orders1)))
    comparison_input.all_orders1 = non_duplicate_ords1
    non_duplicate_ords2 = list(np.unique(np.array(orders2)))
    comparison_input.all_orders2 = non_duplicate_ords2

    report = make_report_comparison(start_datetime, end_datetime, comparison_input, tmp_dir, use_cache)
    report.add_extra_messages(run_errors1 + run_errors2)
    return report


class StrategyBacktestRunInfo(object):
    def __init__(self, strategy_name, strategy_desc, trading_user_id, strategy_code, mnemonic, config_id=None,
                 name=None, extra_params=None, strategy_factory=StrategyFactory):
        self.strategy_name = strategy_name
        self.strategy_desc = strategy_desc
        self.trading_user_id = trading_user_id  # type string
        self.strategy_code = strategy_code
        self.mnemonic = mnemonic
        self.config_id = config_id  # type string
        self.orders = list()
        self.instructions = list()
        self.capitals_serie = None  # dict fetched from data_api or an int
        self.name = name
        self.extra_params = extra_params
        self._strategy_factory = strategy_factory
        if name is None:
            self.name = "{}_{}_{}_{}_{}".format(self.strategy_name, self.strategy_desc, self.strategy_code,
                                                self.mnemonic, self.config_id)

    def load_orders(self, start_date, end_date):
        if self.strategy_name is not None:
            strategy = self.make_strategy_input()
            start_datetime = dt.datetime.combine(start_date, dt.datetime.min.time())
            end_datetime = dt.datetime.combine(end_date, dt.datetime.max.time())
            instructions_backtest, orders_backtest = _load_backtest_orders(
                strategy, start_datetime, end_datetime, self.mnemonic, self.extra_params,
                strategy_factory=self._strategy_factory)

            # Only keep last order status
            orders_backtest = _get_last_backtest_order_status(orders_backtest)
            # 'all' for Analyst is unsupported here for report
            orders_backtest = [o for o in orders_backtest if o['strategy_desc'] != 'all']
            strategy.orders2 = orders_backtest
            self.orders = orders_backtest
            self.instructions = instructions_backtest

    def make_strategy_input(self):
        strategy = StrategyInput(self.strategy_name, self.strategy_desc, self.trading_user_id, self.strategy_code)
        return strategy


def make_report_two_runs(start_date, end_date, strategy_run_info1, strategy_run_info2, use_cache, capital):
    """
    Fetch one or two different backtest runs and create the report.

    :param strategy_run_info1: type StrategyBacktestRunInfo
    :param strategy_run_info2: type StrategyBacktestRunInfo
    :param start_date: type datetime.Date
    :param end_date: type datetime.Date
    :param use_cache: if True use local file cache for fixture_ids and mongo orders
    :return:
    """

    # _add_historical_capital_series(strategies, start_datetime, end_datetime, strategy_factory=strategy_factory)
    start_datetime = dt.datetime.combine(start_date, dt.datetime.min.time())
    end_datetime = dt.datetime.combine(end_date, dt.datetime.max.time())
    tmp_dir = tempfile.mkdtemp(suffix="weekly_report")

    # load orders and instructions
    strategy_run_info1.load_orders(start_date, end_date)
    strategy_run_info2.load_orders(start_date, end_date)

    # a bit hacky: strategy_run_info1 is relative to two different strategies while strategy_input is only one
    strategy_input = strategy_run_info1.make_strategy_input()
    if capital is not None:
        strategy_input.capitals_serie = 10000
    else:
        _add_historical_capital_series([strategy_input], start_datetime, end_datetime
                                       # , strategy_factory=strategy_factory
                                       )

    strategy_input.orders1 = strategy_run_info1.orders
    strategy_input.orders2 = strategy_run_info2.orders
    strategy_input.instructions1 = strategy_run_info1.instructions
    strategy_input.instructions2 = strategy_run_info2.instructions

    comparison_input = ComparisonInput([strategy_input])
    comparison_input.name1 = strategy_run_info1.name
    comparison_input.name2 = strategy_run_info2.name
    comparison_input.all_orders1 = strategy_run_info1.orders
    comparison_input.all_orders2 = strategy_run_info2.orders

    report = make_report_comparison(start_datetime, end_datetime, comparison_input, tmp_dir, use_cache)
    return report


def make_report_comparison(start_datetime, end_datetime, comparison_input, tmp_dir, use_cache):
    """
    Compare two runs of multiple strategies.
    This method can be used to compare two runs generically; it might be useful to compare two version of the same
    strategy in backtest.

    Support overlapped strategies E.g. (football_ess, None) means collecting together
        (football_ess, LTH) and (football_ess, OU)
    Return a Report for a list of StrategyInput.

    :param start_datetime:
    :param end_datetime:
    :param comparison_input: type ComparisonInput
    :param tmp_dir: temporary directory where to store the chart files
    :param use_cache: if True use local file cache for fixture_ids
    :return:
    """
    results = []
    plt.ioff()

    logging.info("Creating daily periods...")
    for strategy in comparison_input.strategies:
        logging.info(strategy.get_short_name())
        logging.info(comparison_input.name1)
        strategy.daily_periods_1 = split_by_time_period(strategy.orders1, 1,
                                                        start_datetime.date(),
                                                        end_datetime.date(),
                                                        use_cache=use_cache)
        logging.info(comparison_input.name2)
        strategy.daily_periods_2 = split_by_time_period(strategy.orders2, 1,
                                                        start_datetime.date(),
                                                        end_datetime.date(),
                                                        use_cache=use_cache)

    logging.info("_get_pnl_distrib_by_period")
    result = _get_pnl_distrib_by_period(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache)
    results.append(result)

    logging.info("_get_daily_roi")
    result = _get_daily_roi(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache)
    results.append(result)

    logging.info("_print_capital")
    result = _print_capital(comparison_input, start_datetime, end_datetime, tmp_dir)
    results.append(result)

    # logging.info("_print_capital_and_missed_volume")
    # result = _print_capital_and_missed_volume(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache)
    # results.append(result)

    # logging.info("_get_weekly_roi")
    # result = _get_weekly_roi(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache)
    # results.append(result)

    logging.info("_get_main_stats")
    result = _get_main_stats(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache)
    results.append(result)

    logging.info("_get_extra_stats")
    result = _get_extra_stats(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache)
    results.append(result)

    # Order by provider, bookmaker and status
    logging.info("_get_order_statuses")
    result = _get_order_statuses(comparison_input, tmp_dir)
    results.append(result)

    logging.info("_get_different_events")
    result = _get_different_events(comparison_input)
    results.append(result)

    logging.info("_get_order_sources")
    result = _get_order_sources(comparison_input)
    results.append(result)

    # result = _price_slippage(comparison_input)
    # results.append(result)
    #
    # result = _price_slippage_by_venue(comparison_input)
    # results.append(result)

    # result = _get_weekly_cumulative_pnl(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache)
    # results.append(result)

    logging.info("_get_daily_cumulative_pnl")
    result = _get_daily_cumulative_pnl(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache)
    results.append(result)

    logging.info("_get_daily_pnl")
    result = _get_daily_pnl(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache)
    results.append(result)

    # logging.info("_get_pnl_per_event")
    # result = _get_pnl_per_event(comparison_input)
    # results.append(result)

    # logging.info("_get_daily_cumulative_vol")
    # result = _get_daily_cumulative_vol(comparison_input, start_datetime, end_datetime, tmp_dir, use_cache)
    # results.append(result)

    # result = _print_orders(comparison_input, start_datetime, end_datetime)
    # results.append(result)
    #
    # result = _get_pnl_zscore(comparison_input, start_datetime, end_datetime, tmp_dir)
    # results.append(result)

    return Report(results, start_datetime, end_datetime)


def compare_two_strategies_and_notify(in_strategy_name1, in_strategy_desc1, in_strategy_code1, in_trading_user_id1,
                                      mnemonic1, config_id1, extra_params1,
                                      in_strategy_name2, in_strategy_desc2, in_strategy_code2, in_trading_user_id2,
                                      mnemonic2, config_id2, extra_params2,
                                      in_start_date, in_end_date, in_capital, use_cache, notify):
    if in_start_date > in_end_date:
        raise ValueError("start_date > end_date {}>{}".format(in_start_date, in_end_date))

    strategy_run_1 = StrategyBacktestRunInfo(in_strategy_name1, in_strategy_desc1, in_trading_user_id1,
                                             in_strategy_code1, mnemonic1, config_id1, extra_params=extra_params1)
    strategy_run_2 = StrategyBacktestRunInfo(in_strategy_name2, in_strategy_desc2, in_trading_user_id2,
                                             in_strategy_code2, mnemonic2, config_id2, extra_params=extra_params2)

    report = make_report_two_runs(in_start_date, in_end_date, strategy_run_1, strategy_run_2, use_cache, in_capital)
    report.send_email(notify)
