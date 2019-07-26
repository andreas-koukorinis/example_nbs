import StringIO
import datetime
import logging
import string
import sys
import tempfile
import traceback
from collections import defaultdict
from copy import deepcopy
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import pytz
from ascii_graph import Pyasciigraph
from jinja2 import Template
from stratagemdataprocessing import data_api
from stratagemdataprocessing.data_api import get_capital_timeseries
from stratagemdataprocessing.enums.odds import Sports
from stratagemdataprocessing.parsing.common.stickers import parse_sticker, sticker_parts_from_sticker
import sgmtradingcore.backtesting.persistence as persistence
from sgmtradingcore.analytics.results.basketball import determine_basketball_outcome_from_api
from sgmtradingcore.analytics.results.common import determine_order_outcome_pnl
from sgmtradingcore.analytics.results.football import determine_football_outcome_from_api
from sgmtradingcore.analytics.results.tennis import determine_tennis_outcome_from_api
from sgmtradingcore.backtesting.backtest_runner import run_backtest_main
from sgmtradingcore.core.notifications import send_trading_system_email
from sgmtradingcore.core.trading_types import OrderStatus
from sgmtradingcore.execution.monitoring import json_to_instruction
from sgmtradingcore.strategies.config.configurations import TRADING_USER_MAP
from sgmtradingcore.strategies.realtime import StrategyFactory
from sgmtradingcore.strategies.strategy_base import StrategyStyle


__author__ = 'lorenzo belli'


def send_single_strategy_backtest_email(strategy_name, strategy_desc, strategy_code, trading_user_id,
                                        start_date, end_date, message, to, mismatch, attachments=None):
    """
    Send an email about a mismatch between production and backtest strategies
    """
    username = TRADING_USER_MAP[trading_user_id]
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    template = Template("""
Automatic backtest for [{{ strategy_name }}, {{ strategy_desc }}, {{ strategy_code }},
{{ username }}, {{ trading_user_id }}]\n
from {{start_date}} to {{end_date}}

<pre>
\n{{ message }}\n
</pre>
        """)
    html_ = template.render(
        strategy_name=strategy_name, strategy_desc=strategy_desc, strategy_code=strategy_code, username=username,
        trading_user_id=trading_user_id, start_date=start_date, end_date=end_date,  message=message)

    if mismatch:
        if start_date == end_date:
            subject = "Backtest MISMATCH {}: {} {} {} {}".format(
                start_date_str, strategy_name, strategy_desc, strategy_code, trading_user_id)
        else:
            subject = "Backtest MISMATCH {}_{}: {} {} {} {}".format(
                start_date_str, end_date_str, strategy_name, strategy_desc, strategy_code, trading_user_id)
    else:
        if start_date == end_date:
            subject = "Backtest match {}: {} {} {} {}".format(
                start_date_str, strategy_name, strategy_desc, strategy_code, trading_user_id)
        else:
            subject = "Backtest match {}_{}: {} {} {} {}".format(
                start_date_str, end_date_str, strategy_name, strategy_desc, strategy_code, trading_user_id)

    send_trading_system_email('', html_, subject, to, files=attachments)


def send_backtest_email(subject, message, notify):
    """
    Send an email about a mismatch between production and backtest strategies
    """
    template = Template("""
Automatic backtesting started and finished:\n

<pre>
Recap is:\n{{ message }}\n
</pre>
        """)
    html_ = template.render(message=message)

    send_trading_system_email('', html_, subject, notify)


def get_strategy_code_from_instruction(instruction):
    """
    Extract the strategy_code of an instruction in production.

    This should be stored in 'details' of the instruction
    :param instruction:
    :return: string or None
    """
    strategy_code = None
    if 'details' in instruction:
        if 'strategy_code' in instruction['details']:
            strategy_code = instruction['details']['strategy_code']
        # Older SIP instructions had no strategy_code in mongo
        if 'strategy_code' not in instruction['details'] and 'set_number' in instruction['details']:
            if instruction['strategy'] == 'tennis_sip_v2_ATP' and instruction['details']['set_number'] == 1:
                strategy_code = 'atp_set1_march2017_new'
            elif instruction['strategy'] == 'tennis_sip_v2_ATP' and instruction['details']['set_number'] == 2:
                strategy_code = 'atp_set2_april2017_new'
            elif instruction['strategy'] == 'tennis_sip_v2_WTA' and instruction['details']['set_number'] == 1:
                strategy_code = 'wta_set1_march2017_new'
            elif instruction['strategy'] == 'tennis_sip_v2_WTA' and instruction['details']['set_number'] == 2:
                strategy_code = 'wta_set2_april2017_new'
            else:
                logging.error("Don't know the strategy_code for instructions {}:\n{} setting to None".format(
                    instruction['id'], instruction))
                strategy_code = None
            logging.warning("Using default strategy_code {} for {} {} instead of real ones".format(
                strategy_code, instruction['strategy'], instruction['strategy_descr']))

        # Older LIP instructions had no strategy_code and no set_number in mongo
        if instruction['strategy'] in ['tennis_lip_ATP', 'tennis_lip_WTA'] and \
                'strategy_code' not in instruction['details'] and \
                'set_number' not in instruction['details'] and\
                'match_score' in instruction['details']:

            set_score = instruction['details']['match_score'][0]  # [Set win player A , set win player B]
            set_n = int(1+set_score[0]+set_score[1])
            strategy_name = instruction['strategy']
            if strategy_name in ['tennis_lip_ATP']:
                strategy_code = 'lip_atp_set' + str(set_n)
            elif strategy_name in ['tennis_lip_WTA']:
                strategy_code = 'lip_wta_set' + str(set_n)
            else:
                raise ValueError("Don't know the strategy_name for instructions {}:\n{}".format(instruction['id'],
                                                                                                instruction))
            logging.warning(
                "Using default strategy_code {} for {} {} instead of real ones".format(
                    strategy_code, instruction['strategy'], instruction['strategy_descr']))

    if 'details' not in instruction and 'tennis' in instruction['strategy']:
        logging.warning("Instruction with no details but for tennis: {}".format(instruction))
    return strategy_code


def get_prod_order_map(start, end, mongo_helper, strategy_name=None, strategy_desc=None, strategy_code=None,
                       trading_user_id=None, use_cache=False):
    """
    Return a map of all the orders and instructions placed between start-delta and end+delta referring to fixture
    starting between start and end.

    :param start: type datetime
    :param end: type datetime
    :param strategy_name:
    :param strategy_desc:
    :param strategy_code:
    :param trading_user_id:
    :param use_cache: use local file cache for production orders in mongo
    :return: {(strategy_name, strategy_desc, trading_user_id, strategy_code):[instructions]},
             {(strategy_name, strategy_desc, trading_user_id):[orders]}
    """

    start_datetime = start-datetime.timedelta(hours=12)
    end_datetime = end+datetime.timedelta(hours=12)
    logging.info("Fetching prod orders and instructions between {} and {}...".format(start_datetime, end_datetime))
    orders = mongo_helper.get_prod_orders_between_datetimes(start_datetime,
                                                            end_datetime,
                                                            strategy_name=strategy_name,
                                                            strategy_desc=strategy_desc,
                                                            trading_user_id=trading_user_id,
                                                            use_cache=use_cache)
    instructions = mongo_helper.get_prod_instructions_between_datetimes(start_datetime,
                                                                        end_datetime,
                                                                        strategy_name=strategy_name,
                                                                        strategy_desc=strategy_desc,
                                                                        strategy_code=strategy_code,
                                                                        trading_user_id=trading_user_id)

    logging.info("Fetched {} orders and {} instructions, now filtering...".format(len(orders), len(instructions)))
    stickers = list(set(i['sticker'] for i in instructions))
    sports = list(set(sticker_parts_from_sticker(s).sport for s in stickers))
    fixtures = persistence.fetch_fixtures_ids(start, end, sports, use_cache=use_cache)
    fixtures_ids = set(f[1] for f in fixtures)

    allowed_instructions = [i for i in instructions if 'sticker' in i and
                            sticker_parts_from_sticker(i['sticker']).scope[1] in fixtures_ids]
    allowed_instructions = [i for i in allowed_instructions if 'strategy' in i]
    allowed_instructions = [i for i in allowed_instructions if str(i['trading_user_id']) in TRADING_USER_MAP.keys()]

    orders = [o for o in orders if ('execution_details' not in o
                                    or o['execution_details']['bookmaker'] not in ['BET365', 'BETCRIS'])]

    allowed_orders = [i for i in orders if str(i['trading_user_id']) in TRADING_USER_MAP.keys()]
    allowed_orders = [i for i in allowed_orders if 'sticker' in i and
                      sticker_parts_from_sticker(i['sticker']).scope[1] in fixtures_ids]
    allowed_orders = [o for o in allowed_orders if 'strategy' in o]

    instructions_map = defaultdict(list)
    instruction_ids_map = defaultdict(list)
    for o in allowed_instructions:
        strategy_code = get_strategy_code_from_instruction(o)
        key = (o['strategy'], o['strategy_descr'], str(o['trading_user_id']), strategy_code)
        instructions_map[key].append(o)
        instruction_ids_map[str(o['id'])] = key

    orders_map = defaultdict(list)
    for o in allowed_orders:
        if 'instruction_id' in o:
            if str(o['instruction_id']) in instruction_ids_map.keys():
                key = instruction_ids_map[str(o['instruction_id'])]
                orders_map[key].append(o)
        else:
            # Cannot recover the strategy_code
            key = (o['strategy'], o['strategy_descr'], str(o['trading_user_id']), None)
            orders_map[key].append(o)

    return instructions_map, orders_map

_THRESHOLD_NUM_SETTLED = 1.8
_THRESHOLD_AVG_PRICE_PERC = 1.3
_THRESHOLD_AVG_PRICE_CONST = 0.15
_THRESHOLD_AVG_PRICE_BY_STICKER_AND_VENUE_PERC = 1.3
_THRESHOLD_AVG_PRICE_BY_STICKER_AND_VENUE_CONST = 0.15
_THRESHOLD_SLIPPAGE_PERC = 2.0
_THRESHOLD_SETTLED_VOLUME_PERC = 1.8
_THRESHOLD_SETTLED_VOLUME_CONST = 5.0  # do not notify differences < 5 pounds
_THRESHOLD_EVENT_PNL_PERC = 1.3
_THRESHOLD_EVENT_PNL_CONST = 5.0  # do not notify differences < 5 pounds
_THRESHOLD_VENUE_PNL_PERC = 1.2
_THRESHOLD_VENUE_PNL_CONST = 5.0  # do not notify differences < 5 pounds
_THRESHOLD_PNL_PERC = 1.15
_THRESHOLD_PNL_CONST = 50.0  # do not notify differences < 50 pounds

_THRESHOLD_RANGE_PNL_PERC = 1.4
_THRESHOLD_RANGE_PNL_CONST = 50.0
_THRESHOLD_MAX_STD_DEV = 80
_THRESHOLD_MAX_ZSCORE_DIFF = 1.0


class TestResult(object):
    def __init__(self, name=None, error_message=None, success_message=None, short_report="", short_name="",
                 good_report_message="", attachments=[]):
        """

        :param name: Name of the test
        :param error_message: text to show in a recap
        :param success_message: text to show in a recap
        :param short_report: short version of the test, to be used in a table
        :param short_name: short version of the test, to be used as label
        :param good_report_message: long message to be reported if there are no errors
        """
        self.test_name = name
        self.success = True
        self.error_message = error_message
        self.success_message = success_message
        self.debug_error_message = None
        self.short_report = short_report  # Used to report a single number about an error
        self.short_name = short_name
        self.good_report_message = good_report_message
        self.attachments = attachments


class BacktestException(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(BacktestException, self).__init__(message)


def _get_size(order):
    if 'exchange_rate' in order and 'currency' in order and order['exchange_rate'] != 0:
        return order['size'] / order['exchange_rate']
    else:
        return order['size']


def _get_size_matched(order):
    if 'exchange_rate' in order and 'currency' in order and order['exchange_rate'] != 0:
        return order['size_matched'] / order['exchange_rate']
    else:
        return order['size_matched']


def _get_starting_capital(strategy_name, strategy_desc, strategy_code, trading_user_id, start_datetime, strategy_factory):
    """
    return the allocated capital at the start od the range, not the full timeseries
    """

    strategy_obj, strategy_class = strategy_factory.create_strategy(strategy_name, strategy_desc, strategy_code,
                                                                   trading_user_id, True, {})
    sport = strategy_obj.get_sport()

    def try_get_capital_timeseries(tr_usr_id, s, str_style, start_bkt, end_bkt, str_name, descr, default):
        try:
            ret = get_capital_timeseries(tr_usr_id, s, str_style, start_bkt, end_bkt, str_name,
                                         strategy_descr=descr)
            return ret
        except Exception:
            return default

    # Get the historical capital time series, updated daily
    capitals_series = {descr: try_get_capital_timeseries(trading_user_id, sport,
                                                         StrategyStyle.to_str(strategy_obj.get_style()),
                                                         start_datetime,
                                                         start_datetime, strategy_name,
                                                         descr, defaultdict(lambda: 0))
                       for descr in [strategy_desc]}

    if len(capitals_series[strategy_desc]) > 1:
        raise ValueError("Too many capitals returned")
    return capitals_series[strategy_desc][capitals_series[strategy_desc].keys()[0]]


def _get_order_id_to_print(o):
    if 'source' in o:
        if o['source'] == 'trading_system':
            return o['instruction_id']
        else:
            return "{:>24}".format(o['source'])
    elif 'instruction_id' in o:
        return o['instruction_id']
    elif 'bet_id' in o:
        return o['bet_id']
    else:
        return "{:>24}".format('unknown')


def _get_orders_debug_print(prod_orders, backtest_orders, extra_backtest_args, strategy_name, strategy_desc,
                            strategy_code, trading_user_id, start_datetime, end_datetime, prod_capital=None,
                            name1='PROD', name2='BACKTEST'):
    """
    Create a StringIO containing a string ot be sent by email to compare backtest and prod orders
    """
    buf = StringIO.StringIO()
    buf.write('----------------------------------------\nDebug information\n\n')

    if prod_capital is None:
        prod_capital = -1
    if 'allocated_capital' in extra_backtest_args.keys():
        backtest_capital = extra_backtest_args['allocated_capital']
    else:
        backtest_capital = prod_capital

    buf.write("{}, starting capital {}\n".format(name1, prod_capital))
    for o in sorted(prod_orders, key=lambda i: (i['sticker'], str(i['placed_time']), i['instruction_id'])):
        price_str = "{}{:.2f}({:.2f})@{:.2f}({:.2f})".format("B" if o['bet_side'] == 'back' else "L", _get_size(o),
                                                             _get_size_matched(o), o['price'],
                                                             o['average_price_matched'])

        buf.write("{}: {:26} {:30} {:>20} {:27}".format(
            _get_order_id_to_print(o), o['sticker'], price_str,
            OrderStatus(o['status']).name, str(o['placed_time'])))
        if 'provider' in o['execution_details']:
            buf.write(" {}_{:6}".format(o['execution_details']['provider'], o['execution_details']['bookmaker']))
        else:
            buf.write(" {}".format(o['execution_details']['bookmaker']))
        if 'sor_codes' in o:
            buf.write(" "+" ".join([c for c in o['sor_codes'].keys() if c != 'sorNetOdds']))
        if o['status'] == OrderStatus.SETTLED:
            buf.write(" {:>8.2f}".format(_get_order_pnl(o)))
        if 'currency' in o and o['currency'] not in ['GBP','']:
            buf.write(" (converted {})".format(o['currency']))
        if 'source' in o and o['source'] != 'trading_system':
            buf.write(" (placed on {})".format(o['source']))

        buf.write("\n")

    buf.write("\n{}, starting capital {}\n".format(name2, backtest_capital))
    for o in sorted(backtest_orders, key=lambda i: (i['sticker'], str(i['placed_time']), i['bet_id'])):
        price_str = "{}{:.2f}({:.2f})@{:.2f}({:.2f})".format("B" if o['bet_side'] == 'back' else "L", _get_size(o),
                                                             _get_size_matched(o), o['price'],
                                                             o['average_price_matched'])

        buf.write("{}: {:26} {:30} {:>20} {:27}".format(
            _get_order_id_to_print(o), o['sticker'], price_str,
            OrderStatus(o['status']).name, str(o['placed_time'])))
        buf.write(" {}".format(o['execution_details']['bookmaker']))
        if 'sor_codes' in o:
            buf.write(" " + " ".join([c for c in o['sor_codes'].keys() if c != 'sorNetOdds']))
        if o['status'] == OrderStatus.SETTLED:
            buf.write(" {:>8.2f}".format(_get_order_pnl(o)))
        if 'currency' in o and o['currency'] != 'GBP' and o['currency'] != '':
            buf.write(" {}".format(o['currency']))

        buf.write("\n")
    return buf.getvalue()


def _get_map_sticker_and_side(orders):
    """
    Return a map {(sticker, side): [orders]}
    """
    orders_map = defaultdict(list)
    for o in orders:
        orders_map[(o['sticker'], o['bet_side'])].append(o)
    return orders_map


def _get_map_sticker_side_and_venue(orders):
    """
    Return a map {(sticker, side, venue): [orders]}
    """
    orders_map = defaultdict(list)
    for o in orders:
        bookmaker = o['execution_details']['bookmaker'].split('_')[-1]
        orders_map[(o['sticker'], o['bet_side'], bookmaker)].append(o)
    return orders_map


def _get_map_per_event(orders):
    """
    Return a map {event: [orders]}
    """
    orders_map = defaultdict(list)
    for o in orders:
        sport, (market_scope, event), market, params, bookmaker = parse_sticker(o['sticker'])
        orders_map[event].append(o)
    return orders_map


def _get_map_per_venue(orders):
    """
    Return a map {bookmaker: [orders]}
    """
    orders_map = defaultdict(list)
    for o in orders:
        bookmaker = o['execution_details']['bookmaker']
        orders_map[bookmaker.split('_')[-1]].append(o)
    return orders_map


def dump_config(obj, nested_level=0, output=sys.stdout):
    """
    Print a dict with newlines
    """
    spacing = '   '
    if type(obj) == dict:
        print >> output, '%s{' % (nested_level * spacing)
        for k, v in sorted(obj.items()):
            if hasattr(v, '__iter__'):
                print >> output, '%s%s:' % ((nested_level + 1) * spacing, k)
                dump_config(v, nested_level + 1, output)
            else:
                print >> output, '%s%s: %s' % ((nested_level + 1) * spacing, k, v)
        print >> output, '%s}' % (nested_level * spacing)
    elif type(obj) == list:
        print >> output, '%s[' % (nested_level * spacing)
        for v in obj:
            if hasattr(v, '__iter__'):
                dump_config(v, nested_level + 1, output)
            else:
                print >> output, '%s%s' % ((nested_level + 1) * spacing, v)
        print >> output, '%s]' % (nested_level * spacing)
    else:
        print >> output, '%s%s' % (nested_level * spacing, obj)


def _test_configs_match(prod_config_id_str, strategy_name, strategy_desc, strategy_code, trading_user_id, mongo_helper,
                        env_config, strategy_factory=StrategyFactory):
    """
    Test if the configuration in production and the one using in backtesting matches

    :param prod_config_id_str: configuration id given by strategy_runs
    """
    prod_config = mongo_helper.get_config_from_id(prod_config_id_str)
    if prod_config['strategy_name'] != strategy_name or prod_config['strategy_desc'] != strategy_desc:
        if prod_config['strategy_name'] not in ['Analyst_FTOUG', 'Analyst_FTAHG'] and \
                        strategy_name not in ['Analyst_FTOUG', 'Analyst_FTAHG'] and \
                        prod_config['strategy_desc'] != 'all':
            raise ValueError("config {} not suitable for {} {}".format(
                prod_config_id_str, strategy_name, strategy_desc))
    _, strategy_class = strategy_factory.create_strategy(strategy_name, strategy_desc, strategy_code, trading_user_id,
                                                        True, env_config)
    backtest_config = strategy_class.get_default_configuration(strategy_name, strategy_desc, strategy_code,
                                                               trading_user_id)

    # Need to store in mongo and retrieve, because in serialization/deserialization some info are lost
    # E.g. GsmCompetitions are not store in Mongo, but Datetime are.
    backtest_config_id = mongo_helper.ensure_configurations('bogus', 'bogus', 'bogus', backtest_config)
    backtest_config_new = mongo_helper.get_config_from_id(backtest_config_id)

    prod_config_str = StringIO.StringIO()
    dump_config(prod_config['params'], output=prod_config_str)
    backtest_config_str = StringIO.StringIO()
    dump_config(backtest_config_new['params'], output=backtest_config_str)
    prod_config_str = prod_config_str.getvalue()
    backtest_config_str = backtest_config_str.getvalue()

    message = "Configs have changed, but we have run the backtest with the old PROD config: "\
              "\nPROD CONFIG {}\n{}\nNEW CONFIG {}\n{}\n".format(
               prod_config_id_str, prod_config_str, backtest_config_id, backtest_config_str)

    res = TestResult(name="Test config match",
                     error_message="Config have changed",
                     success_message="Configs haven't changed in prod",
                     short_report="diff",
                     short_name="configs")
    if prod_config['params'] != backtest_config_new['params']:
        res.success = False
        res.debug_error_message = message
    return res


def _test_different_events(backtest_orders, prod_orders):
    """
    Test if it's betting on different events.
    """
    placed_prod = _get_map_sticker_and_side(prod_orders)
    placed_backtest = _get_map_sticker_and_side(backtest_orders)
    message = ""
    diff_event = ""

    events_prod = [parse_sticker(s[0])[1][1] for s in placed_prod.keys()]
    events_backtest = [parse_sticker(s[0])[1][1] for s in placed_backtest.keys()]
    diff = set(events_prod) ^ set(events_backtest)
    union = set(events_prod + events_backtest)
    if set(events_prod) != set(events_backtest):
        message += "\nBetting on different events: {}/{}\n".format(len(diff), len(union))
        message += "DIFF\n"
        for o in sorted(diff):
            in_prod = o in events_prod
            message += "{}\t{}\n".format(o, 'in prod only' if in_prod else 'in backtest only')
            diff_event = str(o)
            diff_event += ' prod only' if in_prod else ' backtest only'

    res = TestResult(name="Test betting on same events",
                     error_message="Betting on different events \t(e.g. {})".format(diff_event),
                     success_message="Betting on the same events",
                     short_report="{}".format(diff_event),
                     short_name="events")
    if "" != message:
        res.success = False
        res.debug_error_message = message
    return res


def _test_different_stickers(backtest_orders, prod_orders):
    """
    Test if it's betting on different stickers.

    Does not return an error if prod and backtest bet on different events.
    """
    placed_prod = _get_map_sticker_and_side(prod_orders)
    placed_backtest = _get_map_sticker_and_side(backtest_orders)
    message = ""
    diff_sticker = ""

    events_prod = [parse_sticker(s[0])[1][1] for s in placed_prod.keys()]
    events_backtest = [parse_sticker(s[0])[1][1] for s in placed_backtest.keys()]

    placed_prod_filtered = {k: v for k, v in placed_prod.iteritems() if parse_sticker(k[0])[1][1] in events_backtest}
    placed_back_filtered = {k: v for k, v in placed_backtest.iteritems() if parse_sticker(k[0])[1][1] in events_prod}

    if set(placed_prod_filtered.keys()) != set(placed_back_filtered.keys()):
        message += "\nBetting on different stickers\n"

        message += "DIFF\n"
        diff = set(placed_prod_filtered.keys()) ^ set(placed_back_filtered.keys())
        for o in sorted(diff):
            in_prod = o in placed_prod_filtered.keys()
            message += "{:27} {:>2} {}\n".format(
                o[0], 'B' if o[1] == 'back' else 'L', 'in prod only' if in_prod else 'in backtest only')
            diff_sticker = "{} {}".format(o[0], o[1])

    res = TestResult(name="Test betting on same stickers",
                     error_message="Betting on different stickers \t(e.g. {})".format(diff_sticker),
                     success_message="Betting on the same stickers",
                     short_report="{}".format(diff_sticker),
                     short_name="stickers")
    if "" != message:
        res.success = False
        res.debug_error_message = message
    return res


def _make_averaged_price_by_vol(orders):
    tot_volume = 0
    volume_times_price = 0
    for o in orders:
        tot_volume += _get_size_matched(o)
        volume_times_price += _get_size_matched(o)*o['average_price_matched']

    if tot_volume == 0:
        return 0.0
    avg_price = volume_times_price/tot_volume
    return avg_price


def _test_sticker_price_avg_vol(backtest_orders, prod_orders):
    """
    Test that the price averaged by price of settled bets, matches by a threshold.

    Assumes we have settled bets on the same stickers.
    :param backtest_orders:
    :param prod_orders:
    :return:
    """
    settled_backtest = [o for o in backtest_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_prod = [o for o in prod_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_backtest_map = _get_map_sticker_and_side(settled_backtest)
    settled_prod_map = _get_map_sticker_and_side(settled_prod)

    message = 'These stickers have very differents prices (avg by vol):\n'
    mismatch = False
    max_diff = 0
    max_diff_prod = 0
    max_diff_back = 0
    max_diff_sticker = ""

    order_lists = sorted([(k, v) for k, v in settled_backtest_map.iteritems()], key=lambda i: i[0])
    for (sticker, side), backtest_sticker_orders in order_lists:
        try:
            prod_sticker_orders = settled_prod_map[(sticker, side)]
        except KeyError:
            return "{} not present in prod but present in backtest".format((sticker, side))
        price_backtest = _make_averaged_price_by_vol(backtest_sticker_orders)
        price_prod = _make_averaged_price_by_vol(prod_sticker_orders)

        if price_prod == 0 or price_backtest == 0:
            continue
        minimum, maximum = min(price_backtest, price_prod), max(price_backtest, price_prod)
        if maximum-1 > (minimum-1) * _THRESHOLD_AVG_PRICE_PERC and \
                (maximum - minimum) > _THRESHOLD_AVG_PRICE_CONST:
            mismatch = True
            message += "{:26} {:4}: Prod {:5.2f} Vs Backtest {:5.2f}\n".format(
                sticker, side, price_prod, price_backtest)
            if maximum - minimum > max_diff:
                max_diff = maximum - minimum
                max_diff_prod = price_prod
                max_diff_back = price_backtest
                max_diff_sticker = sticker+" "+side

    res = TestResult(name="Test price per sticker, avg by vol",
                     error_message="Prices per sticker, avg by vol, does not match "
                                   "\t(e.g. {} prod/back {:.1f}/{:.1f})".format(
                                    max_diff_sticker, max_diff_prod, max_diff_back),
                     success_message="Prices per sticker, avg by vol, does match",
                     short_report="{}: {:.1f}/{:.1f}".format(max_diff_sticker, max_diff_prod, max_diff_back),
                     short_name="price by st")
    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def _test_sticker_price_avg_vol_and_venue(backtest_orders, prod_orders):
    """
    Test that the prices averaged by volume of settled bets, matches by a threshold.

    Assumes we have settled bets on the same stickers.
    :param backtest_orders:
    :param prod_orders:
    :return:
    """
    settled_backtest = [o for o in backtest_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_prod = [o for o in prod_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_backtest_map = _get_map_sticker_side_and_venue(settled_backtest)
    settled_prod_map = _get_map_sticker_side_and_venue(settled_prod)

    message = 'These stickers and venues have very differents prices (avg by vol):\n'
    mismatch = False
    max_diff = 0
    max_diff_prod = 0
    max_diff_back = 0
    max_diff_sticker = ""

    for bookmaker, backtest_event_orders in settled_prod_map.iteritems():
        if bookmaker not in settled_backtest_map.keys():
            settled_backtest_map[bookmaker] = []

    order_lists = sorted([(k, v) for k, v in settled_backtest_map.iteritems()], key=lambda i: i[0])
    for (sticker, side, venue), backtest_sticker_orders in order_lists:
        try:
            prod_sticker_orders = settled_prod_map[(sticker, side, venue)]
        except KeyError:
            return "{} not present in prod but present in backtest".format((sticker, side, venue))
        price_backtest = _make_averaged_price_by_vol(backtest_sticker_orders)
        price_prod = _make_averaged_price_by_vol(prod_sticker_orders)

        minimum, maximum = min(price_backtest, price_prod), max(price_backtest, price_prod)
        if maximum-1 > (minimum-1) * _THRESHOLD_AVG_PRICE_BY_STICKER_AND_VENUE_PERC and \
                (maximum - minimum) > _THRESHOLD_AVG_PRICE_BY_STICKER_AND_VENUE_CONST:
            mismatch = True
            message += "{:26} {:4} {:4}: Prod {:5.2f} Vs Backtest {:5.2f}\n".format(
                sticker, side, venue, price_prod, price_backtest)
            if maximum - minimum > max_diff:
                max_diff = maximum - minimum
                max_diff_prod = price_prod
                max_diff_back = price_backtest
                max_diff_sticker = sticker+" "+side+" "+venue

    res = TestResult(name="Test price per sticker and venue, avg by vol",
                     error_message="Prices per sticker and venue, avg by vol, does not match"
                                   "\t(e.g. {} prod/back {:.1f}/{:.1f})".format(
                                    max_diff_sticker, max_diff_prod, max_diff_back),
                     success_message="Prices per sticker and venue, avg by vol, does match",
                     short_report="{}: {:.1f}/{:.1f}".format(max_diff_sticker, max_diff_prod, max_diff_back),
                     short_name="price by st, venue")
    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def _get_order_pnl(order):
    if order['status'] != OrderStatus.SETTLED:
        return 0.
    rate = 1.0
    if 'exchange_rate' in order and 'currency' in order and order['exchange_rate'] != 0:
        rate = order['exchange_rate']

    if 'pnl' in order:
        return order['pnl'] / rate
    else:
        if 'outcome' in order.keys():
            return order['outcome']['gross'] / rate
        else:
            return 0.


def _get_pnl(orders):
    tot = 0.0
    for o in orders:
        tot += _get_order_pnl(o)
    return tot


def _test_pnl(backtest_orders, prod_orders):
    """
    Test total pnl
    """
    settled_backtest = [o for o in backtest_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_prod = [o for o in prod_orders if o['status'] in [OrderStatus.SETTLED]]

    message = 'Prod and backtest have very differents pnl:\n'
    mismatch = False

    pnl_backtest = _get_pnl(settled_backtest)
    pnl_prod = _get_pnl(settled_prod)

    minimum, maximum = min(abs(pnl_backtest), abs(pnl_prod)), max(abs(pnl_backtest), abs(pnl_prod))

    if (maximum > minimum * _THRESHOLD_PNL_PERC or pnl_backtest * pnl_prod < 0) and \
            maximum - minimum > _THRESHOLD_PNL_CONST:
        mismatch = True
        message += "Prod {:7.2f} Vs Backtest {:7.2f}\n".format(pnl_prod, pnl_backtest)

    res = TestResult(name="Test total pnl",
                     error_message="Total pnl does not match: \tprod/back {:.1f}/{:.1f}".format(pnl_prod, pnl_backtest),
                     success_message="Total pnl does match: prod/back {:.1f}/{:.1f}".format(pnl_prod, pnl_backtest),
                     short_report="{:.1f}/{:.1f}".format(pnl_prod, pnl_backtest),
                     short_name="pnl")
    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def _test_pnl_per_event(backtest_orders, prod_orders):
    """
    Test that the pnl per event is within a threshold.
    """
    settled_backtest = [o for o in backtest_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_prod = [o for o in prod_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_backtest_map = _get_map_per_event(settled_backtest)
    settled_prod_map = _get_map_per_event(settled_prod)

    max_diff = 0
    max_diff_prod = 0
    max_diff_back = 0
    max_diff_event = ""

    message = 'These events have very differents pnl:\n'
    mismatch = False
    for event, backtest_event_orders in settled_backtest_map.iteritems():
        try:
            prod_event_orders = settled_prod_map[event]
        except KeyError:
            return "{} not present in prod but present in backtest".format(event)

        pnl_backtest = _get_pnl(backtest_event_orders)
        pnl_prod = _get_pnl(prod_event_orders)

        if pnl_backtest == 0 or pnl_prod == 0:
            continue

        minimum, maximum = min(abs(pnl_backtest), abs(pnl_prod)), max(abs(pnl_backtest), abs(pnl_prod))
        if (maximum > minimum * _THRESHOLD_EVENT_PNL_PERC or pnl_backtest * pnl_prod < 0) and \
                maximum - minimum > _THRESHOLD_EVENT_PNL_CONST:
            mismatch = True
            message += "{:14}: Prod {:7.2f} Vs Backtest {:7.2f}\n".format(event, pnl_prod, pnl_backtest)
            if maximum - minimum > max_diff:
                max_diff_prod = pnl_prod
                max_diff_back = pnl_backtest
                max_diff = maximum - minimum
                max_diff_event = str(event)

    res = TestResult(name="Test pnl per event",
                     error_message="pnl per event does not match \t(e.g. {} {:7.1f}/{:7.1f})".format(
                         max_diff_event, max_diff_prod, max_diff_back),
                     success_message="pnl per event does match",
                     short_report="{}: {:.1f}/{:.1f}".format(max_diff_event, max_diff_prod, max_diff_back),
                     short_name="pnl event")
    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def _test_pnl_per_venue(backtest_orders, prod_orders):
    """
    Test that the pnl per venue is within a threshold.
    """
    settled_backtest = [o for o in backtest_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_prod = [o for o in prod_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_backtest_map = _get_map_per_venue(settled_backtest)
    settled_prod_map = _get_map_per_venue(settled_prod)

    max_diff = 0
    max_diff_prod = 0
    max_diff_back = 0
    max_diff_bookmaker = ""

    for bookmaker, backtest_event_orders in settled_prod_map.iteritems():
        if bookmaker not in settled_backtest_map.keys():
            settled_backtest_map[bookmaker] = []

    message = 'These venues have very differents pnl:\n'
    mismatch = False
    for bookmaker, backtest_event_orders in settled_backtest_map.iteritems():
        try:
            prod_event_orders = settled_prod_map[bookmaker]
        except KeyError:
            return "{} not present in prod but present in backtest".format(bookmaker)

        pnl_backtest = _get_pnl(backtest_event_orders)
        pnl_prod = _get_pnl(prod_event_orders)

        minimum, maximum = min(abs(pnl_backtest), abs(pnl_prod)), max(abs(pnl_backtest), abs(pnl_prod))
        if (maximum > minimum * _THRESHOLD_VENUE_PNL_PERC or pnl_backtest * pnl_prod < 0) and \
                maximum - minimum > _THRESHOLD_VENUE_PNL_CONST:
            mismatch = True
            message += "{:7}: Prod {:7.2f} Vs Backtest {:7.2f}\n".format(bookmaker, pnl_prod, pnl_backtest)
            if maximum - minimum > max_diff:
                max_diff_prod = pnl_prod
                max_diff_back = pnl_backtest
                max_diff = maximum - minimum
                max_diff_bookmaker = bookmaker

    res = TestResult(name="Test pnl per venue",
                     error_message="pnl per venue does not match \t(e.g. {} {:7.1f}/{:7.1f})".format(
                         max_diff_bookmaker, max_diff_prod, max_diff_back),
                     success_message="pnl per venue does match",
                     short_report="{}: {:.1f}/{:.1f}".format(max_diff_bookmaker, max_diff_prod, max_diff_back),
                     short_name="pnl venue")
    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def _test_settled_volume_per_sticker(backtest_orders, prod_orders):
    """
    Check that the settled volume per sticker and side matches within a threshold
    """
    settled_backtest = [o for o in backtest_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_prod = [o for o in prod_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_backtest_map = _get_map_sticker_and_side(settled_backtest)
    settled_prod_map = _get_map_sticker_and_side(settled_prod)

    message = 'These stickers have very different settled volumes:\n'
    mismatch = False
    max_diff = 0
    max_diff_prod = 0
    max_diff_back = 0
    max_diff_sticker = ""

    for k, v in settled_backtest_map.iteritems():
        if k not in settled_prod_map.keys():
            settled_prod_map[k] = []

    back_order_lists = sorted([(k, v) for k, v in settled_backtest_map.iteritems()], key=lambda i: i[0][0])
    for (sticker, side), backtest_sticker_orders in back_order_lists:
        prod_sticker_orders = settled_prod_map[(sticker, side)]

        backtest_vol = sum([_get_size_matched(o) for o in backtest_sticker_orders])
        prod_vol = sum([_get_size_matched(o) for o in prod_sticker_orders])
        minimum, maximum = min(backtest_vol, prod_vol), max(backtest_vol, prod_vol)
        if minimum == 0:
            continue
        if maximum > minimum * _THRESHOLD_SETTLED_VOLUME_PERC and maximum > minimum + _THRESHOLD_SETTLED_VOLUME_CONST:
            mismatch = True
            message += "{:26} {:4}: Prod {:>7.2f} Vs Backtest {:>7.2f}\n".format(sticker, side, prod_vol, backtest_vol)
            if maximum - minimum > max_diff:
                max_diff = maximum - minimum
                max_diff_prod = prod_vol
                max_diff_back = backtest_vol
                max_diff_sticker = sticker+" "+side

    res = TestResult(name="Test settled volume per sticker",
                     error_message="Settled volume per sticker is different \t(e.g. {} prod/back {:.1f}/{:.1f})".format(
                         max_diff_sticker, max_diff_prod, max_diff_back),
                     success_message="Settled volume per sticker does match",
                     short_report="{}: {:.1f}/{:.1f}".format(max_diff_sticker, max_diff_prod, max_diff_back),
                     short_name="vol by st")
    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def _test_settled_volume_per_sticker_and_venue(backtest_orders, prod_orders):
    """
    Check that the settled volume per sticker side and venue matches within a threshold.

    Does return an error if a settled volume is 0
    """
    settled_backtest = [o for o in backtest_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_prod = [o for o in prod_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_backtest_map = _get_map_sticker_side_and_venue(settled_backtest)
    settled_prod_map = _get_map_sticker_side_and_venue(settled_prod)

    message = 'These stickers and venues have very different settled volumes:\n'
    mismatch = False
    max_diff = 0
    max_diff_prod = 0
    max_diff_back = 0
    max_diff_sticker = ""

    for k, v in settled_backtest_map.iteritems():
        if k not in settled_prod_map.keys():
            settled_prod_map[k] = []
    for k, v in settled_prod_map.iteritems():
        if k not in settled_backtest_map.keys():
            settled_backtest_map[k] = []

    back_order_lists = sorted([(k, v) for k, v in settled_backtest_map.iteritems()], key=lambda o: o[0][0])
    for (sticker, side, venue), backtest_sticker_orders in back_order_lists:
        prod_sticker_orders = settled_prod_map[(sticker, side, venue)]

        backtest_vol = sum([_get_size_matched(o) for o in backtest_sticker_orders])
        prod_vol = sum([_get_size_matched(o) for o in prod_sticker_orders])
        minimum, maximum = min(backtest_vol, prod_vol), max(backtest_vol, prod_vol)
        # if minimum == 0:
        #     continue
        if maximum > minimum * _THRESHOLD_SETTLED_VOLUME_PERC and maximum > minimum + _THRESHOLD_SETTLED_VOLUME_CONST:
            mismatch = True
            message += "{:26} {:4} {:4}: Prod {:>7.2f} Vs Backtest {:>7.2f}\n".format(
                sticker, side, venue,  prod_vol, backtest_vol)
            if maximum - minimum > max_diff:
                max_diff = maximum - minimum
                max_diff_prod = prod_vol
                max_diff_back = backtest_vol
                max_diff_sticker = sticker+" "+side+" "+venue

    res = TestResult(name="Test settled volume per sticker and venue",
                     error_message="Settled volume per sticker and venue is different"
                                   "\t(e.g. {} prod/back {:.1f}/{:.1f})".format(
                                    max_diff_sticker, max_diff_prod, max_diff_back),
                     success_message="Settled volume per sticker and venue does match",
                     short_report="{}: {:.1f}/{:.1f}".format(max_diff_sticker, max_diff_prod, max_diff_back),
                     short_name="vol by st and venue")
    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def _test_slippage(backtest_orders, prod_orders):
    settled_backtest = [o for o in backtest_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_prod = [o for o in prod_orders if o['status'] in [OrderStatus.SETTLED]]
    settled_backtest_map = _get_map_sticker_and_side(settled_backtest)
    settled_prod_map = _get_map_sticker_and_side(settled_prod)

    message = 'These orders have been matched at a price very different from order\'s price:\n'
    mismatch = False
    max_diff = 0
    max_diff_order = 0
    max_diff_matched = 0
    max_diff_sticker = ""

    message += 'REAL\n'
    for (sticker, side), prod_sticker_orders in settled_prod_map.iteritems():
        for o in prod_sticker_orders:
            if not o['price']/_THRESHOLD_SLIPPAGE_PERC < \
                    o['average_price_matched'] < \
                    o['price'] * _THRESHOLD_SLIPPAGE_PERC:
                mismatch = True
                price_str = "{}{:.2f}({:.2f})@{:.2f}({:.2f})".format("B" if o['bet_side'] == 'back' else "L",
                                                                     _get_size(o),
                                                                     _get_size_matched(o), o['price'],
                                                                     o['average_price_matched'])

                message += "{}: {:26} {}\n".format(o['trade_id'], sticker, price_str)
                diff = abs(o['average_price_matched'] - o['price'])
                if diff > max_diff:
                    max_diff = diff
                    max_diff_order = o['price']
                    max_diff_matched = o['average_price_matched']
                    max_diff_sticker = "{} {}".format(sticker, side)

    message += 'BACKTEST\n'
    for (sticker, side), backtest_sticker_orders in settled_backtest_map.iteritems():
        for o in backtest_sticker_orders:
            if not o['price'] / _THRESHOLD_SLIPPAGE_PERC < \
                    o['average_price_matched'] < \
                    o['price'] * _THRESHOLD_SLIPPAGE_PERC:
                mismatch = True
                price_str = "{}{:.2f}({:.2f})@{:.2f}({:.2f})".format("B" if o['bet_side'] == 'back' else "L",
                                                                     _get_size(o),
                                                                     _get_size_matched(o), o['price'],
                                                                     o['average_price_matched'])

                message += "{}: {:26} {:30}\n".format(o['bet_id'], sticker, price_str)
                diff = abs(o['average_price_matched'] - o['price'])
                if diff > max_diff:
                    max_diff = diff
                    max_diff_order = o['price']
                    max_diff_matched = o['average_price_matched']
                    max_diff_sticker = "{} {}".format(sticker, side)

    res = TestResult(name="Test slippage",
                     error_message="Matched prices are too different from order's prices (e.g. {} {}/{})".format(
                         max_diff_sticker, max_diff_order, max_diff_matched),
                     success_message="Matched prices are the same as order's prices",
                     short_report="{}: {}/{}".format(max_diff_sticker, max_diff_order, max_diff_matched),
                     short_name="slip")

    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + datetime.timedelta(n)


class Period(object):
    def __init__(self, start_datetime, end_datetime):
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.orders = []
        self.fixture_ids_and_time = None  # Fixtures starting in this periods
        self.fixture_ids = None
        self.instructions = []

    def __str__(self):
        return str(self.start_datetime.date())+"_"+str(self.end_datetime.date())

    def add_order(self, order):
        self.orders.append(order)

    def add_instruction(self, instruction):
        self.instructions.append(instruction)

    def is_in_period(self, dt):
        return self.start_datetime <= dt <= self.end_datetime

    def fixture_is_in_period(self, _id):
        return _id in self.fixture_ids

    def get_pnl(self):
        return _get_pnl(self.orders)

    def get_matched_vol(self):
        return sum([o['size_matched'] for o in self.orders if o['status'] == OrderStatus.SETTLED])

    def get_daily_pnl(self):
        ret = dict()
        for date in daterange(self.start_datetime.date(), self.end_datetime.date()):
            valid_fixtures_ids = list()
            for f_id, time in self.fixture_ids_and_time:
                if date == time.date():
                    valid_fixtures_ids.append(f_id)
            valid_orders = list()
            for o in self.orders:
                fixture_id = parse_sticker(o['sticker'])[1][1]
                if fixture_id in valid_fixtures_ids:
                    valid_orders.append(o)
            ret[date] = _get_pnl(valid_orders)
        return ret

    def has_orders(self):
        return len(self.orders) > 0

    def get_missed_vol(self):
        vol_delta = 0.
        pnl_delta = 0.

        football_event_ids = {}
        for instr in self.instructions:
            sticker = instr['sticker']
            sport, market_scope, market, params, _ = parse_sticker(sticker)
            if sport == Sports.FOOTBALL:
                football_event_ids[market_scope[1]] = (-1, -1)

        for k in football_event_ids.keys():
            if k.startswith('GSM'):
                id = int(k[3:])
                football_event_ids[k] = data_api.get_event_outcome(Sports.FOOTBALL, id)
            else:
                raise ValueError("Invalid event id {}".format(k))

        for instr in self.instructions:
            sticker = instr['sticker']
            if instr['size_matched'] < instr['size']:
                remaining = instr['size'] - instr['size_matched']
                if remaining > 0.:
                    print remaining
                vol_delta += remaining

                sport, market_scope, market, params, _ = parse_sticker(sticker)
                if sport == Sports.TENNIS:
                    outcome, n_bet = determine_tennis_outcome_from_api(market, params, market_scope[1])
                elif sport == Sports.BASKETBALL:
                    outcome, n_bet = determine_basketball_outcome_from_api(market, params, market_scope[1])
                elif sport == Sports.FOOTBALL:
                    outcome, n_bet = determine_football_outcome_from_api(market, params, market_scope[1])

                instr_copy = deepcopy(instr)
                instr_copy['ttl'] = "{}".format(instr_copy['ttl'])
                instr_copy['ut'] = "{}".format(instr_copy['ut'])
                instr2 = json_to_instruction(instr_copy)
                instr2.size = remaining
                instr2.matched_amount = 0
                instr2.matched_odds = 0
                instr2.status = -1
                outcome, pnl = determine_order_outcome_pnl(instr2, outcome, n_bet, default_if_unknown=True)

                pnl_delta += pnl
        return vol_delta, pnl_delta


def split_by_time_period(input_orders, num_days, min_date, max_date, use_instructions=False, use_cache=False):
    """
    Split orders in periods, according to fixture start date.

    :param input_orders:
    :param num_days: range to split in
    :param min_date: type datatime.Date
    :param max_date: type datatime.Date
    :return: [Period]
    """

    if min_date > max_date:
        raise ValueError("min_date > max_date: {} > {}", min_date, max_date)
    # ordered_orders = sorted(input_orders, key=lambda o: o['placed_time'])
    # tot_days = (max_day - min_day).days

    periods = []
    day = min_date
    while day <= max_date:
        start_datetime = datetime.datetime.combine(day, datetime.datetime.min.time())
        end_datetime = datetime.datetime.combine(day + datetime.timedelta(days=num_days - 1),
                                                 datetime.datetime.max.time())

        period = Period(start_datetime, end_datetime)
        sports = list(set(sticker_parts_from_sticker(i['sticker']).sport for i in input_orders))
        fixtures = persistence.fetch_fixtures_ids(start_datetime, end_datetime, sports, use_cache)
        fixtures = list(set(fixtures))
        fixture_ids_and_time = [(f_id, pytz.UTC.localize(datetime.datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ'))) for time, f_id in fixtures]
        period.fixture_ids_and_time = fixture_ids_and_time
        period.fixture_ids = [f_id for _, f_id in fixtures]
        periods.append(period)
        day += datetime.timedelta(days=num_days)

    for o in input_orders:
        fixture_id = sticker_parts_from_sticker(o['sticker']).scope[1]
        for period_id in range(len(periods)):
            if periods[period_id].fixture_is_in_period(fixture_id):
                if use_instructions:
                    periods[period_id].add_instruction(o)
                else:
                    periods[period_id].add_order(o)
                continue
    return periods


def _get_periods(backtest_orders, prod_orders, period_in_days, start_date, end_date, use_cache):
    min_date = start_date.date()
    max_date = end_date.date()

    back_periods = split_by_time_period(backtest_orders, period_in_days, min_date, max_date, use_cache)
    prod_periods = split_by_time_period(prod_orders, period_in_days, min_date, max_date, use_cache)
    return back_periods, prod_periods


def _test_pnl_by_time_period(backtest_orders, prod_orders, start_date, end_date, period_in_days, tmp_dir, use_cache):
    """
    Test how the pnl is for each time period
    """

    back_periods, prod_periods = _get_periods(backtest_orders, prod_orders, period_in_days, start_date, end_date,
                                              use_cache)

    if len(back_periods) != len(prod_periods):
        raise ValueError("Different lens {} Vs {}".format(len(back_periods), len(prod_periods)))

    message = 'Pnl per period does not match:\n'
    mismatch = False

    max_diff = 0
    max_diff_prod = 0
    max_diff_back = 0
    max_diff_period = ""
    data_plots = []
    for i in range(len(back_periods)):
        pnl1, pnl2 = prod_periods[i].get_pnl(), back_periods[i].get_pnl()
        diff = abs(pnl1)+abs(pnl2) if pnl1*pnl2 <= 0 else abs(pnl1-pnl2)
        data_plots.append((str(prod_periods[i]), diff))

        maximum, minimum = max(abs(pnl1), abs(pnl2)), min(abs(pnl1), abs(pnl2))
        if (maximum > minimum * _THRESHOLD_RANGE_PNL_PERC or pnl1 * pnl2 < 0) and \
                diff > _THRESHOLD_RANGE_PNL_CONST:

            mismatch = True
            message += "{}: \tProd {:7.0f} Vs Backtest {:7.0f}\n".format(prod_periods[i], pnl1, pnl2)
            if diff > max_diff:
                max_diff_prod = pnl1
                max_diff_back = pnl2
                max_diff = diff
                max_diff_period = str(prod_periods[i])

    graph = Pyasciigraph()
    good_report_message = ""
    for line in graph.graph('PNL diff', data_plots):
        good_report_message += line + "\n"
    message += good_report_message

    labels = [o.start_datetime.strftime('%Y-%m-%d') for o in back_periods]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    prod_pnl = [o.get_pnl() for o in prod_periods]
    back_pnl = [o.get_pnl() for o in back_periods]
    _min = min(prod_pnl+back_pnl+[0])
    _max = max(prod_pnl+back_pnl+[0])
    ax.plot(prod_pnl, 'bo', label='prod pnl', linewidth=2.0)
    ax.plot(back_pnl, 'g^', label='back pnl', linewidth=2.0)
    ax.set_ylim(ymin=_min, ymax=_max)
    plt.xticks(range(len(labels)), labels, rotation=17)
    ax.legend()

    file_path = '%s/%s' % (tmp_dir, "pnl.png")
    plt.savefig(file_path)
    plt.close()

    res = TestResult(name="Test pnl per period",
                     error_message="pnl per period does not match \t(e.g. {:.1f}/{:.1f} {})".format(
                         max_diff_prod, max_diff_back, max_diff_period),
                     success_message="pnl per period does match",
                     short_report="{}: {:.1f}/{:.1f}".format(max_diff_period, max_diff_prod, max_diff_back),
                     short_name="pnl period", good_report_message=good_report_message,
                     attachments=[file_path])
    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def _test_std_dev_by_time_period(backtest_orders, prod_orders, start_date, end_date, period_in_days, use_cache):
    """
    Test the variance and std deviation computed per period

    """

    back_periods, prod_periods = _get_periods(backtest_orders, prod_orders, period_in_days, start_date, end_date,
                                              use_cache)

    if len(back_periods) != len(prod_periods):
        raise ValueError("Different lens {} Vs {}".format(len(back_periods), len(prod_periods)))

    message = 'Std dev on periods too big:\n'
    mismatch = False

    sum_sqr = 0
    num_periods_with_orders = 0
    for i in range(len(back_periods)):
        pnl1, pnl2 = prod_periods[i].get_pnl(), back_periods[i].get_pnl()
        if pnl1 != 0 or pnl2 != 0:
            num_periods_with_orders += 1
        diff = abs(pnl1)+abs(pnl2) if pnl1*pnl2 <= 0 else abs(pnl1-pnl2)
        sum_sqr += diff*diff

    variance = sum_sqr / (num_periods_with_orders - 1) if num_periods_with_orders > 1 else sum_sqr
    std_dev = sqrt(variance)

    if std_dev > _THRESHOLD_MAX_STD_DEV:
        mismatch = True
        message += "{:.1f}>{:.1f}\n".format(std_dev, _THRESHOLD_MAX_STD_DEV)

    good_report_message = "Good Std dev\n{:.1f} < {:.1f}\n".format(std_dev, _THRESHOLD_MAX_STD_DEV)

    res = TestResult(name="Test std dev",
                     error_message="Std dev too big \t\t({:.1f}>{:.1f})".format(std_dev, _THRESHOLD_MAX_STD_DEV),
                     success_message="std dev is {:.1f}<{:.1f}".format(std_dev, _THRESHOLD_MAX_STD_DEV),
                     short_report="{:.1f}>{:.1f}".format(std_dev, _THRESHOLD_MAX_STD_DEV),
                     short_name="Std dev", good_report_message=good_report_message)
    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def _test_zscore_by_time_period(backtest_orders, prod_orders, start_date, end_date, tmp_dir, use_cache):
    """
    Compute zscore daily and draw a graph.

    Only meaningful on longer periods

    """
    period_in_days = 1
    back_periods, prod_periods = _get_periods(backtest_orders, prod_orders, period_in_days, start_date, end_date, use_cache)

    if len(back_periods) != len(prod_periods):
        raise ValueError("Different lens {} Vs {}".format(len(back_periods), len(prod_periods)))

    back_pnls = pd.Series([p.get_pnl() for p in back_periods], index=[p.start_datetime.date() for p in back_periods])
    prod_pnls = pd.Series([p.get_pnl() for p in prod_periods], index=[p.start_datetime.date() for p in prod_periods])
    pnl_df = pd.DataFrame({
        'back_pnl': back_pnls,
        'prod_pnl': prod_pnls,
    })

    zscores = \
        (pnl_df - pnl_df.rolling(window=len(pnl_df), min_periods=1).mean()) / \
        pnl_df.rolling(window=len(pnl_df), min_periods=1).std()

    zscore_diff = zscores['prod_pnl'] - zscores['back_pnl']

    max_diff = 0
    message = 'Zscore on periods too big:\n'
    mismatch = False
    for diff in zscore_diff:
        if abs(diff) > _THRESHOLD_MAX_ZSCORE_DIFF:
            message += "max diff {:.1f}>{:.1f}\n".format(abs(diff), _THRESHOLD_MAX_ZSCORE_DIFF)
            mismatch = True
            if abs(diff) > max_diff:
                max_diff = abs(diff)

    zscores.fillna(0, inplace=True)  # replaces Nan with zeroes

    good_report_message = "Z-Scores\n"

    for k, v in zscores['prod_pnl'].iteritems():
        good_report_message += "{}\t{:6.3f}/{:6.3f}\n".format(k,  v, zscores['back_pnl'][k])
    message += good_report_message

    labels = [o.start_datetime.strftime('%Y-%m-%d') for o in back_periods]
    plt.plot(zscores['prod_pnl'], 'bo', label='prod ZScore')
    plt.plot(zscores['back_pnl'], 'g^', label='back ZScore')
    plt.xticks(range(len(labels)), labels, rotation=17)
    plt.legend()
    file_path = '%s/%s' % (tmp_dir, "zscore.png")
    plt.savefig(file_path)
    plt.close()

    res = TestResult(name="ZScore by period",
                     error_message="ZScore diff too big \t\t({:.1f}>{:.1f})".format(
                         abs(max_diff), _THRESHOLD_MAX_ZSCORE_DIFF),
                     success_message="ZScore diff is {:.1f}<{:.1f}".format(abs(max_diff), _THRESHOLD_MAX_ZSCORE_DIFF),
                     short_report="{:.1f}/{:.1f}".format(abs(max_diff), _THRESHOLD_MAX_ZSCORE_DIFF),
                     short_name="ZScore", good_report_message=good_report_message,
                     attachments=[file_path])
    if mismatch:
        res.success = False
        res.debug_error_message = message
    return res


def compare_orders(strategy_name, strategy_desc, strategy_code, trading_user_id, start_date, end_date,
                   prod_instructions, prod_orders, instructions_backtest, backtest_orders,
                   extra_backtest_args, prod_config_id_str, use_def_config, notify, use_cache,
                   strategy_factory=StrategyFactory, email_footer=""):
    """
    Compare the orders if they match according to some metrics.

    Please note this is more useful to compare few days, or a single day of trading. For statistical analysis or
    long term analysis use something else.
    Send a comparison results emails to every emails in 'notify' plus to the person responsible for the strategy

    :param strategy_name: name of the strategy
    :param strategy_desc: strategy_desc, this is used to specify the allocated capital
    :param strategy_code: is your strategy do noe have one, leave it None
    :param trading_user_id: type string E.g. '562f5bef497aee1c22000001'
    :param start_date: type datetime
    :param end_date: type datetime
    :param prod_instructions: list of instructions as loaded from mongo.
    :param prod_orders: list of orders as loaded from mongo. type is a nested dict where keys are strings
    :param instructions_backtest: list of instructions as loaded from mongo. must have the same structure as prod
    :param backtest_orders: list of orders as loaded from mongo. Must have the same structure as the orders in prod.
    :param extra_backtest_args: if the backtest has been run with extra non-standard parameter, specify it here,
                                or leave it {}
    :param prod_config_id_str: string that identify the configuration used in production
    :param use_def_config: True if the current default configuration has been used for backtesting instead of the
                           historical one
    :param notify: list of email addresses
    :param use_cache: use local file cache for fixtures in mysql
    :param strategy_factory: Class to use to instantiate the strategy object
    :return: True if a mismatch is present
    """
    orders_str = _get_orders_debug_print(prod_orders, backtest_orders, extra_backtest_args,
                                         strategy_name, strategy_desc, strategy_code, trading_user_id,
                                         start_date, end_date)

    tmp_dir = tempfile.mkdtemp(suffix="automatic_backtesting")
    test_results = []

    # Test total pnl
    message = _test_pnl(backtest_orders, prod_orders)
    test_results.append(message)

    # # Std dev per period
    # message = _test_std_dev_by_time_period(backtest_orders, prod_orders, start_date, end_date, 1, use_cache)
    # test_results.append(message)
    #
    # if start_date != end_date:
    #     # Z-score per period
    #     message = _test_zscore_by_time_period(backtest_orders, prod_orders, start_date, end_date, tmp_dir, use_cache)
    #     test_results.append(message)

    # Test which events we have bet on
    message = _test_different_events(backtest_orders, prod_orders)
    test_results.append(message)

    # Test which stickers we have bet on
    message = _test_different_stickers(backtest_orders, prod_orders)
    test_results.append(message)

    # Total pnl per period
    message = _test_pnl_by_time_period(backtest_orders, prod_orders, start_date, end_date, 1, tmp_dir, use_cache)
    test_results.append(message)

    # Test per sticker average price
    message = _test_sticker_price_avg_vol(backtest_orders, prod_orders)
    test_results.append(message)

    # # Test per sticker average price
    # message = _test_sticker_price_avg_vol_and_venue(backtest_orders, prod_orders)
    # test_results.append(message)

    # test per sticker settled volume
    message = _test_settled_volume_per_sticker(backtest_orders, prod_orders)
    test_results.append(message)

    # # test per sticker and venues settled volume
    # message = _test_settled_volume_per_sticker_and_venue(backtest_orders, prod_orders)
    # test_results.append(message)

    # Test per event pnl
    message = _test_pnl_per_event(backtest_orders, prod_orders)
    test_results.append(message)

    # # Test per venue pnl
    # message = _test_pnl_per_venue(backtest_orders, prod_orders)
    # test_results.append(message)

    # Test slippage
    message = _test_slippage(backtest_orders, prod_orders)
    test_results.append(message)

    if not use_def_config and prod_config_id_str is not None:
        # Test that the configuration hasn't changes between prod and backtest
        mongo_helper = persistence.MongoStrategyHelper()
        message = _test_configs_match(prod_config_id_str, strategy_name, strategy_desc, strategy_code, trading_user_id,
                                      mongo_helper, {}, strategy_factory)
        test_results.append(message)

    body = ""
    for r in test_results:
        if r.success:
            body += "GOOD: {}\n".format(r.success_message)
        else:
            body += "BAD:  {}\n".format(r.error_message)
    body += "\n"

    attachments = []
    for t in test_results:
        attachments.extend(t.attachments)
    errors_only = [m for m in test_results if not m.success]

    body += "Details:\n"
    for m in test_results:
        if m.success:
            if m.good_report_message != "":
                body += "------\n" + m.good_report_message + "\n"
        else:
            body += "------\n" + m.debug_error_message + "\n"
    body += orders_str
    _, strategy_class = strategy_factory._get_strategy_object_and_class(strategy_name, strategy_desc, strategy_code, is_backtest=True)

    recipients = strategy_class.get_email_list(strategy_name, strategy_desc)
    # if len(errors_only) == 0:
    #     recipients = []

    body += "\n\n{}".format(email_footer)
    send_single_strategy_backtest_email(strategy_name, strategy_desc, strategy_code, trading_user_id,
                                        start_date, end_date, body,
                                        notify+recipients, len(errors_only) > 0, attachments=attachments)

    return len(errors_only) > 0, test_results


# HELPER FUNCTIONS #
#------------------#

def all_analyst_strategy_descs(names, allowed_trading_users, strategy_factory):
    return all_strategy_descs_from_key(names, 'all', allowed_trading_users, strategy_factory)


def all_ffm_strategy_descs(names):
    from sgmfootball.trading.strategies.factor_model.strategy import FactorModelStrategy
    all_ffm_descs = []
    # get strategy_descr from other versions
    for n in names:
        for sd in FactorModelStrategy.get_valid_strategy_desc(n):
            comps, v = FactorModelStrategy.get_infos_from_desc(sd)
            if len(comps) > 1:
                continue  # take individual ones
            all_ffm_descs.append(sd)
    return all_ffm_descs

def all_strategy_descs_from_key(names, strategy_desc_key, allowed_trading_users, strategy_factory):
    ret = []
    strategy_code = None
    for name in names:
        try:
            strategy_obj, strategy_class = strategy_factory._get_strategy_object_and_class(name, strategy_desc_key, None, is_backtest=True)
        except Exception as e:
            logging.warning(e)
            continue
        for user in allowed_trading_users:
            def_conf = strategy_class.get_default_configuration(name, strategy_desc_key, strategy_code, user)
            strategy_obj.set_configuration(name, strategy_desc_key, None, def_conf, {})
            ret += strategy_obj.strategy_run_ids
    return ret


def _get_runs_in_range(strategy_name, strategy_desc, strategy_code, trading_user_id,
                            start_date, end_date, mongo_helper):
    """
    Returns: first run of a specific strategy that has started before start_date
        and the list of the runs that include any time between start and end
    """
    max_strategy_duration = datetime.timedelta(days=30)
    start_datetime = deepcopy(start_date) - max_strategy_duration
    end_datetime = deepcopy(end_date)
    runs = mongo_helper.get_strategy_runs(strategy_name, strategy_desc, strategy_code, trading_user_id, start_datetime, end_datetime,
                                          mnemonic='prod', is_prod=True, is_optimization=False, is_backtest=False)

    if len(runs) == 0 and strategy_name in ['Analyst_FTOUG', 'Analyst_FTAHG']:
            runs = mongo_helper.get_strategy_runs(strategy_name, 'all', strategy_code, trading_user_id, start_datetime,
                                                  end_datetime,
                                                  mnemonic='prod', is_prod=True, is_optimization=False,
                                                  is_backtest=False)
    if len(runs) == 0 and strategy_name in ['FFM_FTOUG', 'FFM_FTAHG']:
        key_sds = get_ffm_valid_key_strategy_descrs(strategy_name)
        for sd in key_sds:
            runs += mongo_helper.get_strategy_runs(
                strategy_name, sd, strategy_code, trading_user_id, start_datetime, end_datetime,
                mnemonic='prod', is_prod=True, is_optimization=False, is_backtest=False)

    if len(runs) == 0:
        logging.info("No runs found for {} {} {} {} {} "
                     "mnemonic='prod', is_prod=True, is_optimization=False, is_backtest=False".format(
                      strategy_name, strategy_desc, trading_user_id, start_datetime, end_datetime))
        raise ValueError("No runs found between {} and {} for {} {} {} {}".format(start_datetime, end_datetime,
                                                                                  strategy_name, strategy_desc,
                                                                                  strategy_code, trading_user_id))

    runs = sorted(runs, key=lambda x: x['start_time'])
    for i in range(len(runs) - 1):
        # assumes that the end_time is when the next run starts
        runs[i]['end_time'] = runs[i+1]['start_time'] - datetime.timedelta(microseconds=1)
    runs[-1]['end_time'] = runs[-1]['start_time'] + max_strategy_duration

    runs_within_ranges = [r for r in runs if r['start_time'] <= end_datetime and r['end_time'] >= start_date]
    prev = [r for r in runs_within_ranges if r['start_time'] <= start_date]
    prev = sorted(prev, key=lambda x: x['start_time'])
    if len(prev) == 0:
        raise ValueError("No runs started before {}".format(start_date))
    return prev[-1], runs_within_ranges


def _make_error_str(test_results):
    """
    Return a string, representing failed and succeeded tests. For visual purpose.
    """
    message = "["
    for i, e in enumerate(test_results):
        if e.success:
            message += " ,"
        else:
            message += "{},".format(string.ascii_uppercase[i])
    if message[-1] == ",":
        message = message[0:-1]
    message += "]"
    return message


class StrategyComparisonResult(object):
    """
    Handle the comparison result for one strategy only
    """
    def __init__(self, strategy_name, strategy_desc, strategy_code, trading_user_id, test_results=None):
        self.strategy_name = strategy_name
        self.strategy_desc = strategy_desc
        self.trading_user_id = trading_user_id
        self.strategy_code = strategy_code
        if test_results is None:
            test_results = []
        self.test_results = test_results
        self.error_str = None
        self.stacktrace = None
        self.status_str = None
        self.match = True
        self.prod_orders = []
        self.back_orders = []

    def make_one_line(self, verbose=False):
        errors_str = _make_error_str(self.test_results)
        if self.status_str == "SKIPPED":
            errors_str = "......................."
        s = "{:20} {:29} {:29} {:29} {:29} {:9}".format(errors_str, self.strategy_name, self.strategy_desc,
                                                  self.strategy_code, self.trading_user_id,
                                                  self.status_str)

        s += " ords {}/{}".format(len(self.prod_orders), len(self.back_orders))
        if self.status_str == "ERROR":
            s += "\n" + self.error_str + "\n" + self.stacktrace + "\n"
        if verbose:
            for e in self.test_results:
                s += " " + e.short_report + " | "
        return s


class FullComparisonResults(object):
    """
    Handle the comparison of multiple strategies, and send the recap email
    """
    def __init__(self, strategies, start, end):
        """
        :param strategies: [StrategyComparisonResult]
        """
        self.strategies = strategies
        self.footer = ""
        self.simulated_start = start
        self.simulated_end = end
        self._run_start = None
        self._run_end = None
        self.verbose = False

    def _make_email_body(self):
        body = ""
        for s in self.strategies:
            line = s.make_one_line(verbose=self.verbose)
            body += "{}\n".format(line)
        return body

    def _make_error_legend_str(self, errors):
        """
        Return the legend for errors
        """
        message = ""
        for i, e in enumerate(errors):
            message += "{} = {}\n".format(string.ascii_uppercase[i], e.test_name)
        message += "["

        for i, e in enumerate(errors):
            message += string.ascii_uppercase[i] + ','
        if message[-1] == ",":
            message = message[0:-1]
        message += "]"
        if self.verbose:
            message += " "*100
            short_names_legend = self._make_short_name_legend()
            message += short_names_legend
        message += "\n"
        return message

    def _make_short_name_legend(self):
        msg = ""
        for s in self.strategies:
            if len(s.test_results):
                for test in s.test_results:
                    msg += test.short_name + " | "
        return msg

    def _make_test_legend(self):
        for s in self.strategies:
            if len(s.test_results):
                return self._make_error_legend_str(s.test_results)
        return ""

    def _set_footer(self):
        footer = "Backtest Started at {}\n".format(self._run_start)
        footer += "Backtest finished at {}\n".format(self._run_end)
        footer += "Automatic backtest for \n{} to \n{}\n".format(self.simulated_start, self.simulated_end)
        footer += "Analyst and FFM are run with SimpleAlgo"
        self.footer = footer

    def set_running_time(self, run_start, run_end):
        self._run_start = run_start
        self._run_end = run_end

    def send_email(self, notifiees):
        if self.simulated_start == self.simulated_end:
            subject = "Automatic backtesting {}".format(self.simulated_start.strftime('%Y-%m-%d'))
        else:
            subject = "Automatic backtesting {}_{}".format(self.simulated_start.strftime('%Y-%m-%d'),
                                                           self.simulated_end.strftime('%Y-%m-%d'))

        errors_legend = self._make_test_legend()
        body = self._make_email_body()
        content = errors_legend + body + self.footer
        send_backtest_email(subject, content, notifiees)


def auto_backtest_main(in_strategy_name, in_strategy_desc, in_strategy_code, in_trading_user_id, start_date, end_date,
                       use_def_config, repopulate, cmd_line, use_mongo_caches, strategy_factory, framework_providers,
                       mongo_helper, should_have_run, notify, notify_errors,
                       allowed_strategy_names, allowed_strategy_descs, allowed_strategy_codes, allowed_trading_users,
                       extra_backtest_args=None):
    if in_strategy_name is not None:
        allowed_strategy_names = [in_strategy_name]
    if in_strategy_desc is not None:
        allowed_strategy_descs = [in_strategy_desc]
    if in_strategy_code is not None:
        allowed_strategy_codes = [in_strategy_code]
    if in_trading_user_id is not None:
        allowed_trading_users = [in_trading_user_id]

    # special case for Analyst and FFM strategies which use a key word to run multiple competitions
    if in_strategy_desc is None and in_trading_user_id is None and \
            len(set(['Analyst_FTAHG', 'Analyst_FTOUG', 'FFM_FTOUG', 'FFM_FTAHG']).intersection(
                set(allowed_strategy_names))) > 0:
        if in_strategy_name is None:  # get all default strategy names for both FFM and Analyst
            allowed_strategy_descs += all_analyst_strategy_descs(
                ['Analyst_FTAHG', 'Analyst_FTOUG'], allowed_trading_users, strategy_factory)
            allowed_strategy_descs += all_ffm_strategy_descs(['FFM_FTOUG', 'FFM_FTAHG'])
        else:  # add only the strategy descs that are for the provided strategy name if it is for FFM or Analyst
            if in_strategy_name in ['Analyst_FTAHG', 'Analyst_FTOUG']:
                allowed_strategy_descs += all_analyst_strategy_descs(allowed_strategy_names,
                                                                 allowed_trading_users, strategy_factory)
            if in_strategy_name in ['FFM_FTOUG', 'FFM_FTAHG']:
                allowed_strategy_descs += all_ffm_strategy_descs(allowed_strategy_names)

    if start_date is not None and end_date is not None:
        start = start_date
        end = end_date + datetime.timedelta(days=1) - datetime.timedelta(microseconds=1)
    else:
        end = datetime.datetime.combine(datetime.datetime.now(tz=pytz.utc), datetime.datetime.min.time()) \
              - datetime.timedelta(days=3) + datetime.timedelta(days=1)-datetime.timedelta(microseconds=1)
        end = pytz.utc.localize(end)
        start = datetime.datetime(end.year, end.month, end.day, 0, 0, 0, tzinfo=pytz.utc)

    logging.info("Testing between {} and {}".format(start, end))

    orders_map = {}
    instruction_map = {}
    if in_strategy_name is None and in_strategy_desc is None and in_trading_user_id is None:
        for k in should_have_run:
            # Add all the runs that we know should have run, in case they didn't.
            # This detect if a strategy should have bet but didn't
            orders_map[(k[0], k[1], k[2], k[3])] = []
            instruction_map[k] = []

    a, b = get_prod_order_map(start, end, mongo_helper, strategy_name=in_strategy_name, strategy_desc=in_strategy_desc,
                              trading_user_id=in_trading_user_id, use_cache=use_mongo_caches)
    instruction_map.update(a)
    orders_map.update(b)

    runtime_start = datetime.datetime.now()

    mnemonic = 'automatic'
    extra_backtest_args = extra_backtest_args or {}
    extra_backtest_args.update({
        'repopulate': repopulate,
        # 'allocated_capital': 40000, Use constant capital for every backtest
        'use_fixture_cache': use_mongo_caches,
        'use_spark': False,
    })

    range = (start, end, {}, 'test_range_1')
    range_name = range[3]

    if in_strategy_name is not None and in_strategy_desc is not None and in_trading_user_id is not None:
        if (in_strategy_name, in_strategy_desc, in_trading_user_id, in_strategy_code) not in orders_map.keys():
            orders_map[(in_strategy_name, in_strategy_desc, in_trading_user_id, in_strategy_code)] = []
        if (in_strategy_name, in_strategy_desc, in_trading_user_id, in_strategy_code) not in instruction_map.keys():
            instruction_map[(in_strategy_name, in_strategy_desc, in_trading_user_id, in_strategy_code)] = []

    tb = None
    results = []
    keys = sorted(instruction_map.keys())
    for (strategy_name, strategy_desc, trading_user_id, strategy_code) in keys:
        realtime_orders = orders_map.get((strategy_name, strategy_desc, trading_user_id, strategy_code), [])
        realtime_instructions = instruction_map.get((strategy_name, strategy_desc, trading_user_id, strategy_code), [])
        res = StrategyComparisonResult(strategy_name, strategy_desc, strategy_code, trading_user_id)
        results.append(res)
        try:
            if strategy_name not in allowed_strategy_names or \
                            strategy_desc not in allowed_strategy_descs or \
                            strategy_code not in allowed_strategy_codes or \
                            trading_user_id not in allowed_trading_users:
                res.status_str = "SKIPPED"
                logging.info("Skip {:29} {:29} {:29}".format(strategy_name, strategy_desc, trading_user_id))
                continue
            logging.info("Starting comparison for {} {} {} {}".format(strategy_name, strategy_desc, strategy_code, trading_user_id))
            if use_def_config:
                config_id_str = None
                logging.info("Using default config".format())
            else:
                run, all_runs = _get_runs_in_range(strategy_name, strategy_desc, strategy_code, trading_user_id,
                                              start, end, mongo_helper)
                if len(all_runs) > 1:
                    config_ids = list(set([r['config_id'] for r in all_runs]))
                    if len(config_ids) > 1:
                        logging.error('Multiple runs with different config ids could have placed '
                                      'bets between {} and {} ! Arbitrarily taking the first one'.
                                      format(start, end))
                config_id_str = run['config_id']
                if strategy_name in ['Analyst_FTOUG', 'Analyst_FTAHG'] and run['strategy_desc'] == 'all'\
                        and strategy_desc != 'all':
                    # replace the config from key comps as strategy desc by the one with the specific strategy_desc
                    config_id_str = _get_config_id_with_correct_strategy_desc(
                        mongo_helper, config_id_str, strategy_name, strategy_desc, strategy_code)

                if strategy_name in ['FFM_FTOUG', 'FFM_FTAHG']:
                    key_sds = get_ffm_valid_key_strategy_descrs(strategy_name)
                    if run['strategy_desc'] in key_sds and strategy_desc != run['strategy_desc']:
                        # replace the config from key comps as strategy desc by the one with the specific strategy_desc
                        old_config_id_str = config_id_str
                        config_id_str = _get_config_id_with_correct_strategy_desc(
                            mongo_helper, config_id_str, strategy_name, strategy_desc, strategy_code)
                        logging.info('Changed config_id_str from {} in run to {} in backtest after'
                                     ' changing strategy desc'.format(old_config_id_str, config_id_str))
                logging.info("Using strategy config_id {}".format(config_id_str))

            extra_strategy_args = {}
            if strategy_name in ['Analyst_FTOUG', 'Analyst_FTAHG', 'FFM_FTOUG', 'FFM_FTAHG']:
                extra_strategy_args.update({
                    'algo_type': 'SimpleAlgo'})
            if strategy_name in ['Analyst_FTOUG', 'Analyst_FTAHG']:
                extra_strategy_args.update({'is_backtest': True})

            email_footer = "Analyst and FFM are run with 'SimpleAlgo' "

            run_backtest_main(strategy_name, strategy_desc, strategy_code, trading_user_id,
                              extra_strategy_args, extra_backtest_args, config_id_str, mnemonic,
                              cmd_line, {}, strategy_factory, framework_providers, range)
            # Ignore run result and take results from mongo instead

            instructions_backtest, orders_backtest = mongo_helper.get_backtest_result_multiple_days(
                strategy_name, strategy_desc, trading_user_id, strategy_code,
                start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), mnemonic, range_name)

            if len(orders_backtest) == 0:
                logging.info("Backtest returned no orders.")
            else:
                orders_backtest = [o for o in orders_backtest if o['status'] not in [
                    OrderStatus.UNMATCHED, OrderStatus.PARTIALLY_MATCHED, OrderStatus.MATCHED, OrderStatus.PARTIALLY_CANCELLED]]

            has_mismatch = True
            error = None
            try:
                res.prod_orders = realtime_orders
                res.back_orders = orders_backtest
                has_mismatch, error_list = compare_orders(strategy_name, strategy_desc, strategy_code, trading_user_id,
                                                          start, end,
                                                          realtime_instructions, realtime_orders,
                                                          instructions_backtest, orders_backtest,
                                                          extra_backtest_args, config_id_str, use_def_config, notify,
                                                          use_mongo_caches, strategy_factory=strategy_factory,
                                                          email_footer=email_footer)
                res.test_results = error_list
                if has_mismatch:
                    res.match = False

            except Exception as e:
                message = "Could not complete comparison for {} {} {}\n".format(strategy_name, strategy_desc,
                                                                                strategy_code, trading_user_id)
                message += "From {} to {}\n".format(start, end)
                message += "Error is: {}\n".format(e.message)
                tb = traceback.format_exc()
                message += "{}\n".format(tb)
                logging.error(message)
                logging.error("{}\n".format(tb))
                send_single_strategy_backtest_email(strategy_name, strategy_desc, strategy_code, trading_user_id,
                                                    start, end, message, notify_errors, False)
                error = e

            if error is not None:
                res.error_str = error.message
                res.stacktrace = tb
                res.status_str = "ERROR"
            if has_mismatch:
                res.match = False
                res.status_str = "MISMATCH"
                logging.info("MISMATCH, email sent")
            else:
                logging.info("MATCH OK, email sent")
                res.match = True
                res.status_str = "MATCH"
        except Exception as e:
            tb = traceback.format_exc()
            logging.error("{} {} {} {} ERROR: {}\n".format(strategy_name, strategy_desc, strategy_code, trading_user_id, e.message))
            logging.error("{}\n".format(tb))
            res.status_str = "ERROR"
            res.error_str = e.message
            res.stacktrace = tb

            message = "Could not complete comparison for {} {} {} {}\n".format(strategy_name, strategy_desc,
                                                                               strategy_code, trading_user_id)
            message += "From {} to {}\n".format(start, end)
            message += "Error is: {}\n".format(e.message)
            tb = traceback.format_exc()
            message += "{}\n".format(tb)
            send_single_strategy_backtest_email(strategy_name, strategy_desc, strategy_code, trading_user_id, start,
                                                end, message, notify_errors, False)

        logging.info("Comparison finished for {} {} {}\n".format(strategy_name, strategy_desc, trading_user_id))
    logging.info("All comparisons done")

    recap_email = FullComparisonResults(results, start, end)
    recap_email.set_running_time(runtime_start, datetime.datetime.now())
    recap_email.send_email(notify)


def _get_config_id_with_correct_strategy_desc(mongo_helper, config_id_str, strategy_name,
                                              strategy_desc, strategy_code):
    # change the strategy_desc of the config from config_id_str and get the new config id for this updated config
    config = mongo_helper.get_config_from_id(config_id_str)['params']
    config['strategy_desc'] = strategy_desc
    config_id_str = mongo_helper.ensure_configurations(strategy_name, strategy_desc, strategy_code, config)
    return config_id_str


def get_ffm_valid_key_strategy_descrs(strategy_name):
    """
    Returns the strategy desc from FactorModelStrategy that are used to run multiple competitions at the same time
    This is essentially done by getting all the valid strategy desc and checking if this strategy desc includes
    multiples competitions or not
    
    Parameters
    ----------
    strategy_name:  (str) the strategy name, at the moment FFM_FTAHG or FFM_FTOUG

    Returns The list of the valid strategy desc that are used to run multiples competitions
    -------

    """
    from sgmfootball.trading.strategies.factor_model.strategy import FactorModelStrategy
    valid_strategy_descrs = FactorModelStrategy.get_valid_strategy_desc(strategy_name)
    key_sds = []
    for valid_sd in valid_strategy_descrs:
        comps, _ = FactorModelStrategy.get_infos_from_desc(valid_sd)
        if len(comps) > 1:
            key_sds.append(valid_sd)
    return key_sds


def valid_datetime(d):
    return datetime.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
