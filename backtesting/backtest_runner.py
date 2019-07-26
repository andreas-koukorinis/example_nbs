import datetime as dt
import os
import traceback
from collections import defaultdict
from copy import deepcopy, copy
from dateutil import parser
import dill
import numpy as np
import pandas as pd
import pytz
from joblib import Parallel, delayed
import logging
from argparse import ArgumentParser
import re

import sgmtradingcore.backtesting.persistence as persistence
from sgmtradingcore.backtesting.backtest import run_backtest, FrameworkHistoricalProviders
from sgmtradingcore.backtesting.persistence import MongoStrategyHelper
from sgmtradingcore.core.trading_types import OrderStatus
from sgmtradingcore.core.trading_types import InstructionStatus
from sgmtradingcore.core.ts_log import get_logger
from sgmtradingcore.core.ts_log import set_suppress_stdout_flag
from sgmtradingcore.providers.odds_providers import HistoricalFileOddsProvider
from sgmtradingcore.strategies.realtime import StrategyFactory
from sgmtradingcore.strategies.strategy_base import StrategyStyle
from sgmtradingcore.util.errors import joblib_style_error, dump_stack
from sgmtradingcore.util.spark_api import RemoteSparkMapper
from sgmtradingcore.util.script_args import convert_args_list_to_dict
from sgmtradingcore.core.notifications import send_backtest_failure_email, send_backtest_success_email
from stratagemdataprocessing.bookmakers.common.odds.cache import HistoricalCassandraOddsCache
from stratagemdataprocessing.data_api import get_capital_timeseries, get_match_info_single_sport
from stratagemdataprocessing.parsing.common.stickers import sticker_parts_from_sticker
from stratagemdataprocessing.enums.odds import Sports
from sgmtradingcore.strategies.template.strategy import apply_extra_args_to_conf
from sgmtradingcore.strategies.realtime import get_environment
from sgmtradingcore.core.miscellaneous import EmptyProvider
from stratagemdataprocessing.crypto.market.trades_cache import TradesCache
from sgmtradingcore.exchange.trades_matcher.trades_filler import FillingMode
from sgmtradingcore.crypto.asset_inventory import AssetsInventoryManager


__author__ = 'lorenzo belli'


def _serializable_filter_orders(output, spark_input, subrange, backtesting_options):
    """
    Return orders and instructions referring to fixtures started in the valid days of the subrange

    This is serializable as it can be pickled and called from a spark worker
    :param output: as return by run_backtest()
    :param subrange: type Subrange
    :param backtesting_options: type dict
    :return [(update_dt, Order)], [(update_dt, Instruction)]
    """
    start_time = subrange.valid_subrange['start']
    end_time = subrange.valid_subrange['end']

    sports = spark_input.sports
    if Sports.CRYPTO in sports:
        allowed_orders = [i for i in output['orders'].iteritems()]
        allowed_instructions = [i for i in output['instructions'].iteritems()]
        if backtesting_options['last_state_only']:
            raise NotImplementedError("last_state_only not done for crypto")

    else:
        fixtures = persistence.fetch_fixtures_ids(start_time, end_time, sports,
                                                  backtesting_options['use_fixture_cache'])
        fixtures_ids = set(f[1] for f in fixtures)

        if not backtesting_options['store_instructions']:
            allowed_instructions = []
        else:
            allowed_instructions = [i for i in output['instructions'].iteritems() if
                                    sticker_parts_from_sticker(i[1].sticker).scope[1] in fixtures_ids]

        allowed_orders = [i for i in output['orders'].iteritems() if
                          sticker_parts_from_sticker(i[1].sticker).scope[1] in fixtures_ids]

        if backtesting_options['last_state_only']:
            closed_orders = [o for o in allowed_orders if o[1].status in [
                OrderStatus.FAILED, OrderStatus.SETTLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]]

            closed_orders_ids = {o[1].id for o in closed_orders}
            allowed_orders = closed_orders + [o for o in allowed_orders if o[1].id not in closed_orders_ids]

            closed_instructions = [i for i in allowed_instructions if i[1].status in [
                InstructionStatus.CLOSED, InstructionStatus.FAILED]]
            closed_instructions_ids = {i[1].id for i in closed_instructions}
            allowed_instructions = closed_instructions + [i for i in allowed_instructions if
                                                          i[1].id not in closed_instructions_ids]

    return allowed_orders, allowed_instructions


def get_mongo_helper(backtesting_options, sports):
    bo = backtesting_options
    opts_dict = {
        'backtest_db': bo.get('backtest_db', None),
        'realtime_db': bo.get('backtest_db', None),
        'results_coll': bo.get('results_coll', None),
        'runs_coll': bo.get('runs_coll', None),
        'conf_coll': bo.get('conf_coll', None),
        'scratch_coll': bo.get('scratch_coll', None),
        'use_scalable_storage': bo.get('use_scalable_storage', None),
    }
    new_dict = dict()
    for k, v in opts_dict.iteritems():
        if v is not None:
            new_dict[k] = v
    new_dict['sports'] = sports
    if [Sports.CRYPTO] == sports:
        new_dict['backtest_db'] = 'backtesting_crypto'

    return MongoStrategyHelper(**new_dict)


def _serializable_store_output(output, spark_input, check_before_writing=False, logger=None):
    """
    Store the output of the backtest on a subrange

    This is serializable as it can be pickled and called from a spark worker.
    :param output: as return by run_backtest()
    :param spark_input: type SparkInputParams
    :param check_before_writing: if True, fails if the date we tried to store is already in mongo
    """

    filtered_orders, filtered_instructions = _serializable_filter_orders(output,
                                                                         spark_input,
                                                                         spark_input.subrange,
                                                                         spark_input.backtesting_options)

    if logger is not None:
        logger.info(None, "Serializable Storing orders and instructions, {} and {}".format(
            len(filtered_orders), len(filtered_instructions)))

    if spark_input.backtesting_options['store_to_csv']:
        # TODO
        pass

    _, _strategy_class = spark_input.strategy_factory.create_strategy(
        spark_input.strategy_name, spark_input.strategy_desc, spark_input.strategy_code, spark_input.trading_user_id,
        True,
        spark_input.env_config)

    if spark_input.backtesting_options['store_to_db']:
        helper = get_mongo_helper(spark_input.backtesting_options, spark_input.sports)
        try:
            helper.write_backtest_result_to_db(_strategy_class,
                                               spark_input.strategy_run_id,
                                               spark_input.strategy_name,
                                               spark_input.strategy_desc,
                                               spark_input.strategy_code,
                                               spark_input.trading_user_id,
                                               spark_input.mnemonic,
                                               spark_input.strategy_run_start_time,
                                               spark_input.config_id,
                                               filtered_instructions, filtered_orders,
                                               spark_input.subrange.backtest_range,
                                               spark_input.subrange.valid_subrange['start'].date(),
                                               spark_input.subrange.valid_subrange['end'].date(),
                                               spark_input.backtesting_options['use_fixture_cache'],
                                               spark_input.sports,
                                               check_before_writing=check_before_writing)

            signals_key = '%s.signals' % spark_input.strategy_run_id
            logger.info(None, "signals_key: {} ".format(signals_key))
            if 'trading_signals' in output and signals_key in output['trading_signals']:
                logger.info(None, "Writing signals... {}/{}: {} ".format(
                    spark_input.subrange.valid_subrange['start'].date(),
                    spark_input.subrange.valid_subrange['end'].date(),
                    len(output['trading_signals'][signals_key])))

                logger.info(None, "Storing signals...")
                _serializable_store_signals(output['trading_signals'][signals_key].iteritems(),
                                            spark_input,
                                            logger=logger,
                                            mongo_helper=helper)
            else:
                logger.info(None, "No signals to write {}/{}".format(
                    spark_input.subrange.valid_subrange['start'].date(),
                    spark_input.subrange.valid_subrange['end'].date()))

        except Exception as e:
            logger.error(None, "{}".format(e))
            logger.error(None, "{}".format(traceback.format_exc()))
            raise e


def _serializable_store_signals(signals_wrappers, spark_input, logger=None, mongo_helper=None):
    """

    filter signals by event kickoff date and write the remaining to mongo
    :param signals_wrappers: [(timestamp, signal_wrapper)]

    signal_wrapper is like
    {
    'mnemonic': 'lorenzo_test',
    'config_id': '5a7993765f3ced5893d93617',
    'env': 'backtest',
    'strategy_run_id': '5a7994095f3ced5b828e583a'
    'signals': [{'is_back': True, 'name': 'BetSignal', 'event_id': u'ENP2285782',
                 'timestamp': '2016-11-02 17:00:00+00:00', 'sticker': u'BB-EENP2285782-FT12-1.BF',
                 'strategy_code': 'euro_test', 'value': -0.381, 'odds': 1.1, 'strategy_name':
                 'bball_pbp', 'strategy_desc': 'bball_pbp', 'id': '5a79941f5f3ced5b828e583b',
                 'signal_generator_name': 'HistoricalBasketballModelSignalGenerator'},
                {'is_back': True, 'name': 'BetSignal', 'event_id': u'ENP2285783',
                 'timestamp': '2016-11-02 17:00:00+00:00', 'sticker': u'BB-EENP2285783-FTTP-U-158_5.BF',
                 'strategy_code': 'euro_test', 'value': -0.3, 'odds': 1.731585, 'strategy_name':
                 'bball_pbp', 'strategy_desc': 'bball_pbp', 'id': '5a79941f5f3ced5b828e583c',
                 'signal_generator_name': 'HistoricalBasketballModelSignalGenerator'}],
    }
    :return:
    """
    if not spark_input.backtesting_options['store_signals']:
        if logger:
            logger.info(None, "store_signals disabled")
        return
    start_times = dict()  # {event_id: datetime}

    wrappers = [w for w in signals_wrappers]

    start_date = spark_input.subrange.valid_subrange['start'].date()
    end_date = spark_input.subrange.valid_subrange['end'].date()
    start_datetime = dt.datetime.combine(start_date, dt.datetime.min.time()).replace(tzinfo=pytz.UTC)
    end_datetime = dt.datetime.combine(end_date, dt.datetime.max.time()).replace(tzinfo=pytz.UTC)

    new_events_ids = set()
    for placed_dt, wrapper in wrappers:
        new_events_ids.update({signal['event_id'] for signal in wrapper['signals']
                               if signal['event_id'] not in start_times.keys()})

    eid = [int(event_id[3:]) for event_id in new_events_ids]
    for sport in spark_input.sports:
        events = get_match_info_single_sport(sport, eid) or []
        for event in events:
            kick_off = event['start_date']
            start_times[str(event['event_id'])] = parser.parse(kick_off).replace(tzinfo=pytz.utc)

    allowed_wrappers = []
    for placed_dt, wrapper in wrappers:
        wrapper = copy(wrapper)
        wrapper['signals'] = [signal for signal in wrapper['signals']
                              if start_datetime <= start_times[signal['event_id']] <= end_datetime]
        if len(wrapper['signals']):
            allowed_wrappers.append((placed_dt, wrapper))

    del wrappers

    mongo_helper = mongo_helper or get_mongo_helper(spark_input.backtesting_options)
    mongo_helper.write_backtest_template_signals(spark_input.strategy_run_id, spark_input.strategy_name,
                                                 spark_input.strategy_desc, spark_input.strategy_code,
                                                 spark_input.trading_user_id, spark_input.mnemonic,
                                                 spark_input.config_id, allowed_wrappers)


def _serializable_run_backtest_on_subrange(spark_input):
    """
    Run locally the backtest on the subrange specified in spark_input.

    This is serializable as it can be pickled and called from a spark worker.
    :param spark_input: type SparkInputParams
    :return output of run_backtest()
    """
    logger = get_logger(__name__)

    logger.info(None, "[PID {}] Computing subrange {} to {} {} {} {} mnemonic: {} ".format(
        os.getpid(),
        spark_input.subrange.valid_subrange['start'].date(),
        spark_input.subrange.valid_subrange['end'].date(),
        spark_input.strategy_name,
        spark_input.strategy_desc,
        spark_input.strategy_code,
        spark_input.mnemonic,
    ))

    start_time = spark_input.subrange.extended_subrange['start']
    end_time = spark_input.subrange.extended_subrange['end']

    # create a new strategy obj for every run in order to reset it
    _strategy_obj, _strategy_class = spark_input.strategy_factory.create_strategy(
        spark_input.strategy_name, spark_input.strategy_desc, spark_input.strategy_code, spark_input.trading_user_id,
        True, spark_input.env_config)
    _this_range_parameters = deepcopy(spark_input.strategy_parameters)
    _this_range_parameters.update(spark_input.subrange.extra_per_range_params)
    _strategy_obj.set_configuration(spark_input.strategy_name, spark_input.strategy_desc, spark_input.strategy_code,
                                    _this_range_parameters, spark_input.env_config)
    _strategy_obj.set_trading_user_id(spark_input.trading_user_id)

    if spark_input.events is not None:
        _strategy_obj.restrict_events(spark_input.events)

    model_name, model_opt_name, model_config = _strategy_obj.get_model_name_and_opt(
        spark_input.strategy_name, spark_input.strategy_desc, spark_input.strategy_code,
        spark_input.strategy_parameters)
    model_provider = spark_input.framework_providers.model_provider(_strategy_obj.get_sport(),
                                                                    model_name=model_name,
                                                                    model_opt_name=model_opt_name,
                                                                    model_config=model_config)
    action_provider = None
    if spark_input.action_provider_operator is not None:
        action_provider = spark_input.framework_providers.action_provider(
            _strategy_obj.get_sport(),
            delay_ms=spark_input.backtesting_options['action_delay_ms'],
            provider=spark_input.action_provider_operator)

    _strategy_obj.set_storage_params(spark_input.strategy_run_id, spark_input.config_id,
                                     spark_input.mnemonic, None)

    odds_provider = spark_input.framework_providers.market_provider(
        _strategy_obj.get_sport(),
        odds_cache=HistoricalCassandraOddsCache(eager=False),
        expiration_ms=spark_input.backtesting_options['odds_expiration_ms'],
        trades_cache=TradesCache())

    inventory_manager = AssetsInventoryManager(spark_input.asset_inventory)

    output = run_backtest(
        _strategy_obj, start_time, end_time,
        action_provider=action_provider,
        model_provider=model_provider,
        initial_capital=spark_input.subrange.capitals_series,
        inventory_manager=inventory_manager,
        matching_mode=_strategy_class.get_automatic_backtest_matching_mode(),
        odds_provider=odds_provider,
        action_delay_ms=spark_input.backtesting_options['action_delay_ms'],
        framework_providers=spark_input.framework_providers,
        use_trades_matcher=spark_input.backtesting_options['use_trades_matcher'],
        trade_filler_kwargs=spark_input.backtesting_options['trade_filler_kwargs'],
        delay_bets_ms=spark_input.backtesting_options['delay_bets_ms'],
        record_trading_signals=spark_input.backtesting_options['store_signals']
    )
    # After one run the strategy object is no longer usable
    del _strategy_obj
    return output


def _serializable_run_and_store(raw_input, is_multiprocess=False):
    """
    Start a run a backtest on a single subrange, store the results into mongo, return None

    :param raw_input: type SparkInputParams
    """

    logger = get_logger(__name__)
    try:
        _input = dill.loads(raw_input)

        logger.info(None, "[PID {}] Computing subrange {} to {} {} {} {} mnemonic: {} ".format(
            os.getpid(),
            _input.subrange.valid_subrange['start'].date(),
            _input.subrange.valid_subrange['end'].date(),
            _input.strategy_name,
            _input.strategy_desc,
            _input.strategy_code,
            _input.mnemonic,
        ))

        # If it is multiprocess is running on one machine and processes should not compete for I/O
        if is_multiprocess:
            log_file_name = os.path.join(os.environ['HOME'], 'temp_backtest_mp', str(os.getpid()) + '_' + '.txt')
            try:
                os.makedirs(os.path.dirname(log_file_name))
            except Exception:
                pass
            set_suppress_stdout_flag(True, default_log_file=log_file_name, mode='w')

        out = _serializable_run_backtest_on_subrange(_input)

        print "Finished {}_{}".format(_input.subrange.valid_subrange['start'],
                                      _input.subrange.valid_subrange['end'])
        logger.info(None, "Computed {} orders for {}_{} {} {} {} {} ".format(len(out['orders']),
                                                                             _input.subrange.valid_subrange[
                                                                                 'start'].date(),
                                                                             _input.subrange.valid_subrange[
                                                                                 'end'].date(),
                                                                             _input.strategy_name,
                                                                             _input.strategy_desc,
                                                                             _input.strategy_code,
                                                                             _input.mnemonic,
                                                                             )
                    )

        _serializable_store_output(out, _input, check_before_writing=True, logger=logger)
        logger.info(None, "Store finished")
        del out
    except BaseException as e:
        detailed_traceback = joblib_style_error()
        logger.error(None, "{}".format(e))
        logger.error(None, "{}".format(traceback.format_exc()))
        logger.error(None, detailed_traceback)
        raise

    set_suppress_stdout_flag(False)
    return None


class SparkInputParams(object):
    def __init__(self, subrange, strategy_name, strategy_desc, strategy_code, trading_user_id,
                 env_config, strategy_parameters, mnemonic, config_id, backtesting_options, strategy_run_id,
                 strategy_run_start_time, strategy_factory, framework_providers, action_provider_operator, events,
                 sports, asset_inventory):
        """
        :param asset_inventory: {Asset: int of float}
        """
        self.subrange = subrange
        self.strategy_name = strategy_name
        self.strategy_desc = strategy_desc
        self.strategy_code = strategy_code
        self.trading_user_id = trading_user_id
        self.env_config = env_config
        self.strategy_parameters = strategy_parameters
        self.mnemonic = mnemonic
        self.config_id = config_id
        self.backtesting_options = backtesting_options
        self.strategy_run_id = strategy_run_id
        self.strategy_run_start_time = strategy_run_start_time
        self.strategy_factory = strategy_factory
        self.framework_providers = framework_providers
        self.action_provider_operator = action_provider_operator
        self.events = events
        self.sports = sports
        self.asset_inventory = asset_inventory


class Subrange(object):
    def __init__(self, backtest_range, extended_subrange, valid_subrange, extra_per_range_params, capitals_series,
                 backtesting_options):
        """

        :param backtest_range: the full backtest range
                               as returned by RunnableSimpleStrategyUnit.backtesting_date_ranges(). type is tuple of 4
        :param valid_subrange: valid days for which this wubrange has run for
        :param extended_subrange: valid_subrange extended with 'days_before' and 'days_after'. dd
                                  type {'start': datetime, 'end': datetime'}
        :param extra_per_range_params: strategy parameter to add to every subrange run for this range. type dict
        :param capitals_series: capital to use while running for this subrange. type dit or int.
                                type {'start': datetime, 'end': datetime'}

        """

        self.backtest_range = backtest_range
        self.extended_subrange = self._ensure_datetime(extended_subrange)
        self.valid_subrange = self._ensure_datetime(valid_subrange)
        self.extra_per_range_params = extra_per_range_params
        self.capitals_series = capitals_series
        self.backtesting_options = backtesting_options

    @staticmethod
    def _ensure_datetime(d):
        for k, v in d.iteritems():
            if isinstance(v, pd.Timestamp):
                d[k] = v.to_datetime()
            assert isinstance(d[k], dt.datetime)
        return d


class StrategyBacktestRunner(object):
    """
    Class to perform a single backtest run, i.e. a strategy run on a list of ranges.

    """

    def __init__(self, strategy_name, strategy_desc, strategy_code, trading_user_id, strategy_factory, mnemonic,
                 mongo_helper, extra_backtesting_options, extra_strategy_args,
                 is_optimization, command_line, env_config, framework_providers, config_id_str=None,
                 action_provider_operator=None, events=None):
        """
        Create the runner and log the run and the config in mongo.

        It creates its own strategy_objects, for every range/subrange that will be run.
        Default strategy parameters or from mongo (config_id_str) will be applied first.
        extra_strategy_args will be applied next.
        Default parameters + extra_strategy_args will be logged. Extra per-range parameters will be applied
        but not logged.

        :param strategy_name: name of the strategy
        :param strategy_desc: desc of the strategy
        :param extra_strategy_args: non default strategies argument to be applied on top of others params
        :param extra_backtesting_options: dict with options for the backtesting, like 'repopulate'.
            See _get_default_backtesting_options() for the available options
        :param command_line: cmd line used to run this backtest. For logging purpose
        :param config_id_str: _id of the mongo doc representing the strategy config to use.
                              If None, default will be used. if present, extra_strategy_args will not be applied
        :return:
        """

        self._logger = get_logger(__name__)

        strategy_obj, strategy_class = strategy_factory.create_strategy(strategy_name, strategy_desc, strategy_code,
                                                             trading_user_id, True, env_config, events=events)

        self._banned_days = strategy_class.get_banned_days(strategy_name, strategy_desc, trading_user_id, strategy_code)

        self._all_days_to_run = []

        self._strategy_name = strategy_name
        self._strategy_desc = strategy_desc
        self._strategy_code = strategy_code
        self._trading_user_id = trading_user_id
        self._strategy_obj = None
        self._strategy_class = strategy_class
        self._strategy_factory = strategy_factory
        self._framework_providers = framework_providers
        self._mnemonic = mnemonic
        self._action_provider_operator = action_provider_operator

        self._backtesting_options = self._get_default_backtesting_options()
        self._backtesting_options.update(extra_backtesting_options)
        self._sports = [strategy_obj.get_sport()]

        self._events = events

        if mongo_helper is None:
            mongo_helper = get_mongo_helper(self._backtesting_options, self._sports)
        else:
            if self._backtesting_options['backtest_db'] is not None:
                raise ValueError("You asked for a non standard backtest_db but also gave your own MongoHelper; "
                                 "are you sure?")
        self._mongo_helper = mongo_helper

        self._start_times = dict()  # {event_id: datetime}

        self._env_config = env_config
        self._command_line = command_line

        if config_id_str is not None:
            if extra_strategy_args != {}:
                # raise ValueError("when using config_id_str  cannot use extra_strategy_args")
                self._logger.error(
                    None, "when using config_id_str cannot use extra_strategy_args. Both have been applied here")
            self._logger.info(None, "Using strategy config_id {}".format(config_id_str))
            param_doc = mongo_helper.get_and_verify_config_from_id(config_id_str, strategy_name, strategy_desc)
            self._strategy_parameters = param_doc['params']
            if ('strategy_code' in self._strategy_parameters and
                self._strategy_parameters['strategy_code'] != strategy_code) or \
                    ('strategy_code' in param_doc and
                     param_doc['strategy_code'] != strategy_code):
                raise ValueError("Strategy_code in config {} is not {}. {}".format(config_id_str, strategy_code,
                                                                                   param_doc))
        else:
            self._logger.info(None, "Using default strategy config")
            self._strategy_parameters = strategy_class.get_default_configuration(strategy_name, strategy_desc,
                                                                                 strategy_code, trading_user_id)
            # Strategy default param plus backtest extra params. Per-range params will be added later.

        # extra_strategy_args
        if 'is_template_strategy' not in self._strategy_parameters or \
                not self._strategy_parameters['is_template_strategy']:
            self._strategy_parameters.update(extra_strategy_args)
        else:
            self._strategy_parameters = apply_extra_args_to_conf(self._strategy_parameters, extra_strategy_args)

        # Store config_id
        self._config_id = self._mongo_helper.ensure_configurations(strategy_name, strategy_desc, strategy_code,
                                                                   self._strategy_parameters)
        self._logger.info(None, "CONFIG_ID is {}".format(self._config_id))

        self._this_range_parameters = None
        self._strategy_start_time = dt.datetime.now(tz=pytz.utc)

        # Store the run
        env = 'backtest'
        #         self._trading_user_id
        self._strategy_run_id = self._mongo_helper.log_strategy_run(
            strategy_name, strategy_desc, strategy_code, trading_user_id, self._config_id, self._strategy_start_time,
            self._mnemonic, False, True, is_optimization, command_line, env, backtest_config=self._backtesting_options)
        self._logger.info(None, "STRATEGY_RUN_ID is {}".format(self._strategy_run_id))

    @staticmethod
    def _get_default_backtesting_options():
        """
        :return: default set of options for the backtest framework
        """
        return {'repopulate': False,  # If False, only run the backtest for the days not present in the db,
                # if True delete every range requested to run, then run and write again.
                'save_n_days': 7,  # backtesting and saving to db is done in batched on save_n_days days
                'days_before': 1,  # How many days of backtesting to start running before the start_date.
                #   Gives time to place orders
                'days_after': 2,  # Days of backtesting to run after the end_date. Ensures orders get settled.
                'hours_before': 0,  # How many days of backtesting to start running before the start_date.
                'hours_after': 0,  # Days of backtesting to run after the end_date. Ensures orders get settled.
                'store_to_db': True,  # If True store backtests results to mongo
                'backtest_db': None,  # if not None use a specific db for MongoStrategyHelper backtest
                'results_coll': None,  # if backtest_db is not None use a specific results_coll for MongoStrategyHelper
                'runs_coll': None,  # if backtest_db is not None use a specific runs_coll for MongoStrategyHelper
                'conf_coll': None,  # if backtest_db is not None use a specific conf_coll for MongoStrategyHelper
                'scratch_coll': None,  # if backtest_db is not None use a specific scratch_coll for MongoStrategyHelper
                'use_scalable_storage': True,  # Put orders and instructions in separate mongo collections
                'store_to_csv': False,  # If True store backtests results to local csv file
                'csv_folder': '',  # Location to write the csv output file

                # Constant allocated capital for all the simulation date range. If None
                # the historical allocated capital timeseries is used instead
                'allocated_capital': None,

                # Starting inventory, type {Asset: float or int}
                'inventory': None,

                # Used to make queries faster, it is not an error to keep it as it is
                'sports': [Sports.FOOTBALL, Sports.TENNIS, Sports.BASKETBALL],
                
                'use_fixture_cache': False,  # Use locally cached method to fetch fixtures data
                'run_banned_days': False,
                # delay action data by this amount
                'action_delay_ms': 0,
                'use_spark': False,
                'spark_master_host': 'spark://176.227.211.162:7077',  # math01.stratagem.co
                'spark_executor_memory': '8g',
                'launcher_environment': 'stratagem-trading-dev',
                'spark_driver_memory': '4g',
                'spark_app_name': None,
                'max_spark_cores': 30,
                'use_multiprocess': False,
                'n_workers': 4,  # for multiprocessing only
                'store_instructions': True,  # Deactivate if you are generating too many bets for one single day
                'last_state_only': False,  # only store the last state of every order and instruction. Not recommended
                # since it might hide bugs
                'delete_only': False,  # Delete the old backtest result (like -repopulate) but does not rerun

                # If true, log Template Strategy Signals
                # if 'store_signals' and 'repopulate' are both True, it repopulate the signals
                'store_signals': False,
                # suspend every sticker after this delay
                'odds_expiration_ms': 0,

                # Delay placement of every bets
                'delay_bets_ms': 0,

                # New alternative to the exchange sim
                'use_trades_matcher': False,
                # Input of TradesFiller
                'trade_filler_kwargs': {
                    'filling_mode': FillingMode.FAVG1,
                    'seconds_forward': 20}
                }

    def _days_already_run(self, range_name):
        """
        Return dates for which this strategy configuration has already been run.

        Query is based on strategy_name, strategy_desc, mnemonic, range_name. @see get_days_already_run()
        :param range_name: name of the range, as given by RunnableSimpleStrategyUnit.backtesting_date_ranges()[0][3]
        :return a list of dates ['2015-08-12', ...]
        """

        days_already_run = self._mongo_helper.get_days_already_run(self._strategy_name,
                                                                   self._strategy_desc,
                                                                   self._trading_user_id,
                                                                   self._strategy_code,
                                                                   self._mnemonic,
                                                                   range_name)
        days_already_run = [d for d in days_already_run if
                            range_name[0].date() <=
                            pytz.UTC.localize(dt.datetime.strptime(d, "%Y-%m-%d")).date()
                            <= range_name[1].date()]

        self._logger.info(None, "already run {}/{} days".format(len(set(days_already_run)),
                                                                (range_name[1].date() - range_name[0].date()).days + 1))
        if len(days_already_run) != len(set(days_already_run)):
            raise RuntimeError("Multiple days returned for {} {} {} {} {}".format(
                self._config_id, self._strategy_name, self._strategy_desc, self._mnemonic, range_name))

        return days_already_run

    def _get_days_to_run(self, start_date, end_date, backtest_range):
        """
        Split the range between start_date and end_date in multiple ranges, skipping days already run and banned days.

        Ignores days_after and days_before
        :return a list of dict [{'start': '2015-08-12', 'end': '2015-08-14'}, ...]
        """

        daily_ranges_to_run = []
        days_already_run = self._days_already_run(backtest_range)

        if self._backtesting_options['repopulate']:
            days_already_run = []

        if self._backtesting_options['delete_only']:
            return []

        all_days_to_run_range = pd.date_range(start=start_date, end=end_date, freq='D')
        all_days_to_run_range = [dt.datetime.strftime(date, '%Y-%m-%d') for date in all_days_to_run_range]
        if self._backtesting_options['run_banned_days']:
            days_to_run_range = list(set(all_days_to_run_range)
                                     - set(days_already_run))
        else:
            days_to_run_range = list(set(all_days_to_run_range)
                                     - set(days_already_run)
                                     - set(self._banned_days))
        days_to_run_range.sort()
        days_to_run_range = np.array([dt.datetime.strptime(el, '%Y-%m-%d') for el in days_to_run_range])

        if len(days_to_run_range) > 0:
            temp = {'start': dt.datetime.strftime(days_to_run_range[0], '%Y-%m-%d')}
            # divide days range in subranges in case we have holes due to blacklisted days
            for i, day in enumerate(days_to_run_range[:-1]):
                if days_to_run_range[i + 1] - days_to_run_range[i] > dt.timedelta(days=1):
                    temp['end'] = dt.datetime.strftime(days_to_run_range[i], '%Y-%m-%d')
                    daily_ranges_to_run.append(deepcopy(temp))
                    temp = {'start': dt.datetime.strftime(days_to_run_range[i + 1], '%Y-%m-%d')}
            temp['end'] = dt.datetime.strftime(days_to_run_range[-1], '%Y-%m-%d')
            daily_ranges_to_run.append(deepcopy(temp))

        return daily_ranges_to_run

    def run_and_log_backtest(self, in_backtest_range):
        """
        Run the strategy backtest in a range.

        All the bets places among these two dates will be kept. days_before and days_after will be added here.
        Results are stored in mongo and/or csv file.
        :param in_backtest_range: a range as returned by RunnableSimpleStrategyUnit.backtesting_date_ranges()
        :return list of update_time and orders placed in within the backtest_range. Does not include orders already
            present in mongo. update_time can be outside the ranges.
            return type is [(datetime, Order)]
        """

        backtest_range = deepcopy(in_backtest_range)
        backtest_range = (backtest_range[0],
                          pytz.UTC.localize(dt.datetime.combine(backtest_range[1], dt.datetime.max.time())),
                          backtest_range[2],
                          backtest_range[3]
                          )
        start_date, end_date = backtest_range[0], backtest_range[1]
        extra_per_range_params, range_name = backtest_range[2], backtest_range[3]
        if start_date > end_date:
            raise ValueError("End date before start date")

        # check if the stop (not included) is at least 2 days before today
        max_end_day = dt.datetime.today().replace(tzinfo=pytz.utc) - dt.timedelta(days=2)
        if end_date > max_end_day:
            self._logger.warn(None,
                              "WARNING: end date {} too close, might not have time to settle bets".format(end_date))

        if self._backtesting_options['repopulate'] or self._backtesting_options['delete_only']:
            self._logger.info(None, "Repopulating the db from {} to {} for {} {} {} {} {}".format(
                start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'),
                self._strategy_name, self._strategy_desc, self._trading_user_id, self._strategy_code, self._mnemonic))

            self._mongo_helper.delete_backtest_results(self._strategy_name, self._strategy_desc,
                                                       self._trading_user_id, self._strategy_code, self._mnemonic,
                                                       start_date.strftime('%Y-%m-%d'),
                                                       end_date.strftime('%Y-%m-%d'),
                                                       range_name)

            if self._backtesting_options['store_signals']:
                self._mongo_helper.delete_signals(self._strategy_name,
                                                  self._strategy_desc,
                                                  self._trading_user_id,
                                                  self._strategy_code,
                                                  self._mnemonic,
                                                  start_date=start_date.date(),
                                                  end_date=end_date.date())

        if self._backtesting_options['delete_only']:
            self._logger.info(None, "delete_only is active, not running")
        daily_ranges_to_run = self._get_days_to_run(
            start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), backtest_range)
        if len(daily_ranges_to_run) == 0:
            self._logger.info(None, "All days already run")
            return []

        self._logger.info(None,
                          "---- Starting range {} to {}".format(start_date.strftime('%Y-%m-%d'),
                                                                end_date.strftime('%Y-%m-%d')))

        subranges_to_run = []
        for daily_range in daily_ranges_to_run:

            # print '------- Running range from {} to {}'.format(daily_range['start'], daily_range['end'])
            daily_subranges = split_ranges_no_extra_days(daily_range, self._backtesting_options['save_n_days'])
            for idx, range_split in enumerate(daily_subranges):
                # add days_before and days_after
                valid_subrange = {'start': range_split['start'],
                                  'end': range_split['end'] + dt.timedelta(days=1) - dt.timedelta(milliseconds=1)
                                  }
                extended_subrange = {'start': valid_subrange['start'] -
                                     dt.timedelta(days=self._backtesting_options['days_before'],
                                                  hours=self._backtesting_options['hours_before']),
                                     'end': valid_subrange['end'] +
                                     dt.timedelta(days=self._backtesting_options['days_after'],
                                                  hours=self._backtesting_options['hours_after'])}

                self._logger.info(None,
                                  "Creating subrange {}/{} from {} to {} ({} to {})({} valid days +{}/+{})".format(
                                      idx, len(daily_subranges),
                                      extended_subrange['start'].date(), extended_subrange['end'].date(),
                                      valid_subrange['start'].date(), valid_subrange['end'].date(),
                                      (valid_subrange['end'].date() - valid_subrange['start'].date()).days + 1,
                                      self._backtesting_options['days_before'], self._backtesting_options['days_after']
                                  ))

                capitals_series = None
                if Sports.CRYPTO not in self._sports:
                    capitals_series = self._get_timeseries(extended_subrange['start'], extended_subrange['end'])
                subranges_to_run.append(Subrange(backtest_range, extended_subrange, valid_subrange,
                                                 extra_per_range_params, capitals_series, self._backtesting_options))

        if self._backtesting_options['use_spark']:
            if self._events is not None:
                raise ValueError('Event Filtering only works locally on a single core')
            self._run_subranges_on_spark_store_remotely(subranges_to_run)
        elif self._backtesting_options['use_multiprocess']:
            if self._events is not None:
                raise ValueError('Event Filtering only works locally on a single core')
            self._run_subranges_locally_multiprocess(subranges_to_run)
        else:
            self._run_subranges_locally(subranges_to_run)

        self._logger.info(None, "STRATEGY_RUN_ID is {}".format(self._strategy_run_id))
        self._logger.info(None, "CONFIG_ID is {}".format(self._config_id))
        self._logger.info(None, "COMMAND LINE is {}".format(self._command_line))
        return

    def _run_subranges_locally(self, subranges_to_run):
        """
        Run locally and store into mongo the results for all the subrange.

        Run locally and sequentially.
        In one range fails some data will be already written in mongo.
        :param subranges_to_run: type is [(extended_range, extra_per_range_params, capitals_series, backtest_range,
                                         range_split)]
        :return:
        """
        for s in subranges_to_run:
            self._logger.info(None, "--------------- {} {} {} {}".format(
                self._strategy_name, self._strategy_desc, self._strategy_code, self._trading_user_id
            ))
            self._logger.info(None,
                              "--------------- Running subrange from {} to {} ({} to {})({} valid days +{}/+{})".format(
                                  s.extended_subrange['start'].date(), s.extended_subrange['end'].date(),
                                  s.valid_subrange['start'].date(), s.valid_subrange['end'].date(),
                                  (s.valid_subrange['end'].date() - s.valid_subrange['start'].date()).days + 1,
                                  self._backtesting_options['days_before'], self._backtesting_options['days_after'],
                              ))
            out = self.run_backtest_on_range_locally(s)

            spark_input = self._make_spark_input(s)
            _serializable_store_output(out, spark_input, check_before_writing=True, logger=self._logger)
            del out
        return

    def _run_locally_and_store_subrange(self, subrange):

        out = self.run_backtest_on_range_locally(subrange)
        spark_input = self._make_spark_input(subrange)
        _serializable_store_output(out, spark_input, check_before_writing=True, logger=self._logger)
        del out

    def _make_spark_input(self, subrange):
        asset_inventory = self._backtesting_options['inventory']
        return SparkInputParams(subrange, self._strategy_name, self._strategy_desc, self._strategy_code,
                                self._trading_user_id,
                                self._env_config, self._strategy_parameters, self._mnemonic, self._config_id,
                                self._backtesting_options,
                                self._strategy_run_id, dt.datetime.now(tz=pytz.utc),
                                self._strategy_factory, self._framework_providers,
                                self._action_provider_operator, self._events, self._sports,
                                asset_inventory
                                )

    def _run_subranges_locally_multiprocess(self, subranges_to_run):
        """
        Run locally and store into mongo the results for all the subrange.

        Run locally and sequentially.
        In one range fails some data will be already written in mongo.
        :param subranges_to_run: type is [(extended_range, extra_per_range_params, capitals_series, backtest_range,
                                         range_split)]
        :return:
        """

        self._logger.info(None, "Starting pool with {} processes".format(self._backtesting_options['n_workers']))

        spark_inputs = []
        for subrange in subranges_to_run:
            spark_input = self._make_spark_input(subrange)
            spark_inputs.append(spark_input)

        # For multiprocessing, one probably does not need the dill/undill but the callback is shared with spark
        inputs_array = [dill.dumps(si) for si in spark_inputs]
        self._logger.info(None, "Starting on subranges {}".format(subranges_to_run))

        Parallel(n_jobs=self._backtesting_options['n_workers'])(delayed(_serializable_run_and_store)(
            raw_input=si, is_multiprocess=True)
                                                                for si in inputs_array)

        self._logger.info(None, "All subranges done")

    def _run_subranges_on_spark_store_remotely(self, subranges_to_run):
        """
        Every spark job run a single subrange and store the results in Mongo.

        This version is not completely safe because spark guarantee 'At Least One execution', and since out jobs
        hae side effects (writing to DB) this is not completely safe. Spark is not intended for jobs with side effects.

        TO get around this we could run all on spark, wait for results.
        :param subranges_to_run: type is [(extended_range, extra_per_range_params, capitals_series, backtest_range,
                                         range_split)]
        :return:
        """
        mapper = RemoteSparkMapper(self._backtesting_options['devpi_user'], self._backtesting_options['devpi_index'],
                                   self._backtesting_options['launcher_environment'],
                                   min(self._backtesting_options['max_spark_cores'], len(subranges_to_run)),
                                   self._backtesting_options['spark_master_host'],
                                   executor_memory=self._backtesting_options['spark_executor_memory'],
                                   max_failures=1,  # If one single task fails, stop the full application
                                   driver_memory=self._backtesting_options['spark_driver_memory'],
                                   app_name=self._backtesting_options['spark_app_name'],
                                   pool='backtest', reuse_workers=False)
        # mapper = LocalSparkMapper(self._backtesting_options['devpi_user'], self._backtesting_options['devpi_index'],
        #                           'stratagem-trading-dev',
        #                           len(subranges_to_run), 'spark://10.1.4.207:7077',  # lorenzo-pc 10.1.4.207
        #                           executor_memory='4gb', max_failures=1, driver_memory='4g',
        #                           app_name=app_name)
        self._logger.info(None, "Created spark mapper")

        spark_inputs = []
        for subrange in subranges_to_run:
            spark_input = self._make_spark_input(subrange)
            spark_inputs.append(spark_input)

        try:
            self._logger.info(None, "Starting {} jobs".format(len(spark_inputs)))
            results = mapper.map([dill.dumps(i) for i in spark_inputs], _serializable_run_and_store,
                                 number_partition=len(spark_inputs))
            res = results

        except persistence.MongoDuplicateDayException as e:
            self._logger.error(None, "Exception occurred while running on spark, "
                                     "We tried to store data for a run which already existed. {}".format(e))
            raise e
        except Exception as e:
            self._logger.error(None, "Exception occurred while running on spark, "
                                     "some data might have been written into DB already. {}".format(e))
            tb = traceback.format_exc()
            message = "{}\n".format(tb)
            self._logger.error(None, message)
            raise

        return res

    def _get_timeseries(self, start_datetime, end_datetime):
        self._strategy_obj, self._strategy_class = self._strategy_factory.create_strategy(self._strategy_name,
                                                                                          self._strategy_desc,
                                                                                          self._strategy_code,
                                                                                          self._trading_user_id,
                                                                                          True, self._env_config,
                                                                                          events=self._events)
        sport = self._strategy_obj.get_sport()
        trading_user_id = self._trading_user_id
        constant_capital = self._backtesting_options['allocated_capital']
        if constant_capital is None:
            def try_get_capital_timeseries(tr_usr_id, sport_, str_style, start_bkt, end_bkt, str_name, descr, default):
                try:
                    ret = get_capital_timeseries(tr_usr_id, sport_, str_style, start_bkt, end_bkt, str_name,
                                                 strategy_descr=descr)
                    return ret
                except Exception:
                    return default

            # Get the historical capital time series, updated daily
            capitals_series = {descr: try_get_capital_timeseries(trading_user_id, sport,
                                                                 StrategyStyle.to_str(self._strategy_obj.get_style()),
                                                                 start_datetime,
                                                                 end_datetime, self._strategy_name,
                                                                 descr, defaultdict(lambda: 0))
                               for descr in self._strategy_obj.strategy_run_ids}
        else:

            capitals_series = {descr: defaultdict(lambda: constant_capital)
                               for descr in self._strategy_obj.strategy_run_ids}
        return capitals_series

    def _filter_orders(self, output, subrange):
        """
        Return orders and instructions referring to fixtures started in the valid days of the subrange
        :return [(update_dt, Order)], [(update_dt, Instruction)]
        """

        return _serializable_filter_orders(output, subrange, self._backtesting_options)

    # def _store_signals(self, signals_wrappers, start_date, end_date):
    #     """
    #
    #     filter signals by event kickoff date and write the remaining to mongo
    #     :param signals_wrappers: [(timestamp, signal_wrapper)]
    #
    #     signal_wrapper is like
    #     {
    #     'mnemonic': 'lorenzo_test',
    #     'config_id': '5a7993765f3ced5893d93617',
    #     'env': 'backtest',
    #     'strategy_run_id': '5a7994095f3ced5b828e583a'
    #     'signals': [{'is_back': True, 'name': 'BetSignal', 'event_id': u'ENP2285782',
    #                  'timestamp': '2016-11-02 17:00:00+00:00', 'sticker': u'BB-EENP2285782-FT12-1.BF',
    #                  'strategy_code': 'euro_test', 'value': -0.381, 'odds': 1.1, 'strategy_name':
    #                  'bball_pbp', 'strategy_desc': 'bball_pbp', 'id': '5a79941f5f3ced5b828e583b',
    #                  'signal_generator_name': 'HistoricalBasketballModelSignalGenerator'},
    #                 {'is_back': True, 'name': 'BetSignal', 'event_id': u'ENP2285783',
    #                  'timestamp': '2016-11-02 17:00:00+00:00', 'sticker': u'BB-EENP2285783-FTTP-U-158_5.BF',
    #                  'strategy_code': 'euro_test', 'value': -0.3, 'odds': 1.731585, 'strategy_name':
    #                  'bball_pbp', 'strategy_desc': 'bball_pbp', 'id': '5a79941f5f3ced5b828e583c',
    #                  'signal_generator_name': 'HistoricalBasketballModelSignalGenerator'}],
    #     }
    #     :return:
    #     """
    #     if not self._backtesting_options['store_signals']:
    #         self._logger.info(None, "store_signals disabled")
    #         return
    #
    #     wrappers = [w for w in signals_wrappers]
    #
    #     start_datetime = dt.datetime.combine(start_date, dt.datetime.min.time()).replace(tzinfo=pytz.UTC)
    #     end_datetime = dt.datetime.combine(end_date, dt.datetime.max.time()).replace(tzinfo=pytz.UTC)
    #
    #     new_events_ids = set()
    #     for placed_dt, wrapper in wrappers:
    #         new_events_ids.update({signal['event_id'] for signal in wrapper['signals']
    #                                if signal['event_id'] not in self._start_times.keys()})
    #
    #     eid = [int(event_id[3:]) for event_id in new_events_ids]
    #     for sport in self._sports:
    #         events = get_match_info_single_sport(sport, eid) or []
    #         for event in events:
    #             kick_off = event['start_date']
    #             self._start_times[str(event['event_id'])] = parser.parse(kick_off).replace(tzinfo=pytz.utc)
    #
    #     allowed_wrappers = []
    #     for placed_dt, wrapper in wrappers:
    #         wrapper = copy(wrapper)
    #         wrapper['signals'] = [signal for signal in wrapper['signals']
    #                               if start_datetime <= self._start_times[signal['event_id']] <= end_datetime]
    #         if len(wrapper['signals']):
    #             allowed_wrappers.append((placed_dt, wrapper))
    #
    #     del wrappers
    #
    #     self._mongo_helper.write_backtest_template_signals(self._strategy_run_id, self._strategy_name,
    #                                                        self._strategy_desc, self._strategy_code,
    #                                                        self._trading_user_id, self._mnemonic, self._config_id,
    #                                                        allowed_wrappers)

    def run_backtest_on_range_locally(self, subrange):
        """
        Run locally the backtest on the specified range, with some extra parameters.
        subrange.extra_per_range_params: parameters to be applied to this range only.
                                       Will be applied on top of the strategy parameters.
        subrange.capitals_series: can be both a capital series or a number
        :return run_backtest() output
        """

        spark_input = self._make_spark_input(subrange)
        return _serializable_run_backtest_on_subrange(spark_input)


def split_ranges_no_extra_days(in_range, n_days_split):
    """
    Split a range in subranges of n_days_split days.

    The returned subranges never overlaps. Ignores days_before and days_after.
    :param in_range: must have type {'start: '2016-12-31', 'end': '2017-12-31'}
    :param n_days_split: days of every subranges to create
    :return utc localized datetime with time 00:00
        return type is [{'start': datetime, 'end': datetime}, ...]
    """
    if n_days_split <= 0:
        raise ValueError("Invalid n_days_split: {}".format(n_days_split))
    start_datetime = pytz.utc.localize(dt.datetime.strptime(in_range['start'], '%Y-%m-%d'))
    end_datetime = pytz.utc.localize(dt.datetime.strptime(in_range['end'], '%Y-%m-%d'))

    save_n_days = dt.timedelta(days=n_days_split)
    if end_datetime < start_datetime:
        raise ValueError("range has only invalid days")
    temp = [t.to_datetime() for t in pd.date_range(start=start_datetime,
                                                   end=end_datetime,
                                                   freq=save_n_days)]

    subranges = []
    if len(temp) > 0:
        for el in temp:
            end_day = el + save_n_days - dt.timedelta(days=1)
            rg = {'start': el,
                  'end': min(end_day, end_datetime)}
            if el <= pytz.utc.localize(dt.datetime.strptime(in_range['end'], '%Y-%m-%d')):
                subranges.append(rg.copy())

    return subranges


def _apply_extra_strategy_args_to_conf(strategy_def_config, extra_strategy_args):
    """
    Apply extra strategy arguments; works both with templated and non templated strategies
    :param strategy_def_config: type {}
    :param extra_strategy_args: type {}
    :return:
    """
    if 'is_template_strategy' not in strategy_def_config or not strategy_def_config['is_template_strategy']:
        strategy_def_config = copy(strategy_def_config)
        strategy_def_config.update(extra_strategy_args)
    else:
        strategy_def_config = apply_extra_args_to_conf(strategy_def_config, extra_strategy_args)
    return strategy_def_config


def _verify_extra_params(strategy_name, strategy_desc, strategy_code, trading_user_id, extra_strategy_args,
                         strategy_factory):
    """
    Verify that the extra parameters are allowed for this strategy
    """
    strategy_obj, strategy_class = strategy_factory.create_strategy(strategy_name, strategy_desc, strategy_code,
                                                                    trading_user_id, True,
                                                                    {})
    strategy_def_config = strategy_class.get_default_configuration(strategy_name, strategy_desc, strategy_code,
                                                                   trading_user_id)

    # extra_strategy_args
    strategy_def_config = _apply_extra_strategy_args_to_conf(strategy_def_config, extra_strategy_args)

    strategy_obj.set_configuration(strategy_name, strategy_desc, strategy_code, strategy_def_config, {})
    applied_conf = strategy_obj.get_current_configuration()
    strategy_factory.config_dict_match(strategy_def_config, applied_conf)

    # Test that range names have no duplicates
    ranges = strategy_class.backtesting_date_ranges(strategy_name, strategy_desc)
    names = {}
    for r in ranges:
        if r[3] in ['', None]:
            raise ValueError("Range cannot have null name")
        if r[3] not in names.keys():
            names[r[3]] = 1
        else:
            raise ValueError("Range with name {} is present twice".format(r[3]))

    return applied_conf


def is_automatic_backtest_mnemonic(mnemonic):
    return re.search(r"testing_[0-9a-zA-Z]+$", mnemonic) is not None


def run_backtest_main(strategy_name, strategy_desc, strategy_code, trading_user_id, extra_strategy_args,
                      backtest_runner_args, config_id_str, mnemonic, cmd_line, env_config,
                      strategy_factory=StrategyFactory, framework_providers=FrameworkHistoricalProviders, range=None,
                      action_provider_operator=None, events=None):
    """
    Run a strategy backtest on all the backtest ranges fot that strategy.

    Results are written in mongodb.

    :param strategy_name:
    :param strategy_desc:
    :param strategy_code:
    :param trading_user_id:
    :param extra_strategy_args: params to be applied to the strategy
    :param backtest_runner_args: args for StrategyBacktestRunner
    :param config_id_str: if not None this config will be applied instead
    :param mnemonic:
    :param cmd_line:
    :param range: if present use this range instead. Format is (start_dt, end_dt, {extra_args_per_range}, range_name)
    :param events: if not None will only run backtest for these events. Should be a space-separated list of ENET / GSM
    ids. Subsequent re-runs should be with -repopulate if all events are required
    :return: list of backtest return values
    """

    _, strategy_class = strategy_factory.create_strategy(strategy_name, strategy_desc, strategy_code, trading_user_id,
                                                         True, env_config, events=events)

    if events is not None:
        logging.warning('Running with event filtering - be sure to use -repopulate for subsequent runs')

    if strategy_name in ['Analyst_FTOUG', 'Analyst_FTAHG']:
        extra_strategy_args['is_backtest'] = True

    mongo_helper = None
    _verify_extra_params(strategy_name, strategy_desc, strategy_code, trading_user_id, extra_strategy_args,
                         strategy_factory)

    runner = StrategyBacktestRunner(strategy_name, strategy_desc, strategy_code, trading_user_id, strategy_factory,
                                    mnemonic, mongo_helper, backtest_runner_args, extra_strategy_args, False, cmd_line,
                                    env_config, framework_providers, config_id_str, action_provider_operator,
                                    events=events)

    if range is not None:
        ranges = [range]
    else:
        ranges = strategy_class.backtesting_date_ranges(strategy_name, strategy_desc)

    for _, backtest_range in enumerate(ranges):
        start_datetime = pytz.UTC.localize(dt.datetime.combine(backtest_range[0].date(),
                                                               dt.datetime.min.time()))
        end_datetime = pytz.UTC.localize(dt.datetime.combine(backtest_range[1].date(),
                                                             dt.datetime.max.time()))
        runner.run_and_log_backtest((start_datetime, end_datetime, backtest_range[2], backtest_range[3]))

    return runner._config_id


def _valid_datetime(d):
    """
    Convert input string into datetime
    """
    return dt.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=pytz.UTC)


def _get_extra_strategy_args_for_signal_replay(strategy_name, strategy_desc, strategy_code, env_name, mnemonic,
                                               strategy_factory):
    """
    Works by replacing all the signal generators in the original strateg with a single ReplaySignalGenerator
    which act as all of the replaced generators.
    """

    trading_user_id = get_environment(env_name)['trading_user_id']

    _, strategy_class = strategy_factory._get_strategy_object_and_class(strategy_name, strategy_desc, strategy_code, True)
    original_strategy_config = strategy_class.get_default_configuration(
        strategy_name, strategy_desc, strategy_code, trading_user_id)

    extra_strategy_args = {
        'new_signals_generators': [{
            'component_name': 'ReplaySignalGenerator',
            'config_name': 'default',
            'config': {
                'strategy_name': strategy_name,
                'strategy_desc': strategy_desc,
                'strategy_code': strategy_code,
                'env': env_name,
                'mnemonic': mnemonic,
                'signals_generators': original_strategy_config['signals_generators'],
            }
        }]
    }
    return extra_strategy_args


def _get_extra_strategy_args_for_basic_execution(strategy_name, strategy_desc, strategy_code, env_name, mnemonic,
                                                 strategy_factory, trading_user_id):

    strategy_obj, strategy_class = strategy_factory.create_strategy(strategy_name, strategy_desc, strategy_code,
                                                                    trading_user_id, True,
                                                                    {})
    sport = strategy_obj.get_sport()

    if sport == Sports.CRYPTO:
        extra_strategy_args = {}
    else:

        extra_strategy_args = {
            'new_trade_executors': [{'component_name': 'PlainTradeExecutor',
                                     'config': {},
                                     'config_name': 'default'},
                                    {'component_name': 'CloseTradeExecutorSimple',
                                     'config': {},
                                     'config_name': 'default'}]
        }
    return extra_strategy_args


def run_backtest_main_notify(strategy_name, strategy_descr, strategy_code, trading_user_id, notify,
                             strategy_args, strategy_config_id, mnemonic, start_date, end_date, backtest_runner_args,
                             cmd_line, strategy_factory, framework_providers, env, action_provider_operator=None,
                             replay_signals=False, replay_mnemonic=None, events=None):
    """
    Starts the backtest and send an email in case of success or failure
    """
    if replay_signals:
        logging.info("Replaying signals, extra_strategy_args will be ignored")

        if env is None:
            raise ValueError("env cannot be NONE")

        strategy_args = _get_extra_strategy_args_for_signal_replay(strategy_name, strategy_descr, strategy_code,
                                                                   env, replay_mnemonic, strategy_factory)

    if backtest_runner_args['use_trades_matcher']:
        strategy_args = _get_extra_strategy_args_for_basic_execution(strategy_name, strategy_descr, strategy_code,
                                                                     env, mnemonic, strategy_factory, trading_user_id)

    try:
        range = None
        if start_date is not None and end_date is not None:
            range = (start_date, end_date, {}, 'range_name1')
        env_config = {} if env is None else get_environment(env)
        run_backtest_main(strategy_name, strategy_descr, strategy_code, trading_user_id,
                          strategy_args, backtest_runner_args, strategy_config_id, mnemonic, cmd_line,
                          env_config, strategy_factory, framework_providers, range, action_provider_operator,
                          events=events)
    except BaseException:
        tb = traceback.format_exc()
        logger = logging.getLogger(__name__)
        logger.error(tb)
        detailed_traceback = joblib_style_error()

        logger.error(detailed_traceback)
        if notify is not None:
            send_backtest_failure_email(strategy_name, strategy_descr, detailed_traceback, notify)

        dump_stack(logger, [strategy_name, strategy_descr])
        raise
    else:
        if notify is not None:
            send_backtest_success_email(strategy_name, strategy_descr, notify)


def create_arguments(input_args=None):
    parser = ArgumentParser(description='Run a strategy backtest, optionally on Spark. Results are in Mongo.')
    parser.add_argument(
        '-strategy_name', help='Universal strategy name', required=True)
    parser.add_argument(
        '-strategy_desc', help='Strategy description', required=True)
    parser.add_argument(
        '-strategy_code', help='Strategy code', required=False, default=None)
    parser.add_argument('-env', help='Specify bookmaker account id. DEV, BETA, PROD, PROD_ALGO_SPORTS, ...',
                        required=False, default=None)
    parser.add_argument(
        '-spark', action='store_true', default=False, help='Run backtest on Spark')
    parser.add_argument(
        '-notify', help='Email addresses to alert of failure or success', required=False, default=None)
    parser.add_argument(
        '-strategy_args', help='Add strategy specific args e.g. arg1=val1', required=False, nargs='*')
    parser.add_argument(
        '-config_id', help='Mongo ObjectID of the configuration, optional.', required=False, default=None)
    parser.add_argument(
        '-repopulate',
        help='If you want to completely repopulate the db for this date range or just extend it',
        action='store_true', default=False)
    parser.add_argument('-mnemonic', default='', required=True,
                        help="Mnemonic name to give to this run, 'automatic' and 'report' are reserved")
    parser.add_argument('-start_date', help='E.g. 2016-12-31', required=False, default=None, type=_valid_datetime)
    parser.add_argument('-end_date', help='E.g. 2016-12-31', required=False, default=None, type=_valid_datetime)
    parser.add_argument('-trading_user_id', help='Dynamic capital allocation will be used for this '
                                                 '[strategy_name, strategy_desc, trading_user_id]',
                        required=True, default=None)
    parser.add_argument('-capital', help='Constant capital allocation. Replace -trading_user_id',
                        required=False, default=None, type=int)
    parser.add_argument('-cache', help='use file cache for mongo', action='store_true', required=False, default=False)
    parser.add_argument('-use_spark', help='if you want to run on spark', action='store_true', required=False,
                        default=False)
    parser.add_argument('-launcher_environment', help='the launcher environment for the spark workers', required=False,
                        default='stratagem-trading-dev')

    parser.add_argument('-u', help='devpi_user for spark', required=False, default=None, type=str)
    parser.add_argument('-i', help='devpi_index for spark', required=False, default=None, type=str)
    parser.add_argument('-use_multiprocess', help='if you want to run locally multiprocessing', action='store_true',
                        required=False, default=False)
    parser.add_argument('-max_spark_cores', help='num of spark cores to use at most', required=False,
                        default=52, type=int)
    parser.add_argument('-n_workers', help='num of worker processes', required=False, default=0, type=int)
    parser.add_argument('-no_instructions', help='if you do not want to store instructions', action='store_true',
                        required=False, default=False)
    parser.add_argument('-last_state', help='only store the last state of orders and instructions', action='store_true',
                        required=False, default=False)
    parser.add_argument('-save_n_days', help='number of days per run', required=False, default=7, type=int)
    parser.add_argument('-days_prior', help='number of days prior', required=False, default=1, type=int)
    parser.add_argument('-days_post', help='number of days post', required=False, default=1, type=int)
    parser.add_argument('-hours_prior', help='number of hours prior', required=False, default=0, type=int)
    parser.add_argument('-hours_post', help='number of hours post', required=False, default=0, type=int)
    parser.add_argument('-use_scalable_storage', help='Always true, deprecated',
                        action='store_true', required=False, default=True)
    parser.add_argument('-no_scalable_storage', help='This is ignored and always true',
                        action='store_true', required=False, default=False)
    parser.add_argument('-backtest_db', help='override the default database, e.f. backtesting or backtesting_dev',
                        required=False, default=None, type=str)
    parser.add_argument('-results_coll', help='use a non default result collection',
                        required=False, default=None, type=str)
    parser.add_argument('-delete_only', help='Delete the old backtest result (like -repopulate) but does not run',
                        action='store_true', required=False, default=False)
    parser.add_argument('-replay_signals', help='Fetch the previously run signal and replay them'
                                                '(WARNING: this change the strategy config)',
                        action='store_true', required=False, default=False)
    parser.add_argument('-replay_mnemonic', help='mnemonic of the strategy being replayed. use `prod` for production',
                        required=False, type=str)
    parser.add_argument('-action_provider', default=None, required=False,
                        help="select a non standard action provider")
    parser.add_argument('-odds_expiration_ms', help='expire odds ticks', required=False, default=0, type=int)
    parser.add_argument('-events', default=None, required=False, nargs='+',
                        help="Restrict strategy run to listed events")
    parser.add_argument('-delay_bets_ms', help='Delay placement of every transaction',
                        required=False, default=0, type=int)
    parser.add_argument('-store_signals', help='store strategy signals in mongo and delete old ones',
                        action='store_true', required=False, default=False)
    parser.add_argument('-use_trades_matcher', help='Use TradesMatcher instead of ExSim', action='store_true',
                        required=False, default=False)

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    if args.mnemonic in ['automatic', 'report'] or is_automatic_backtest_mnemonic(args.mnemonic):
        raise ValueError("This mnemonic is not allowed: {}".format(args.mnemonic))
    if args.use_spark and None in [args.u, args.i]:
        raise ValueError("With spark you have to specify -u -i")
    if (args.u is not None or args.i is not None) and args.use_spark is False:
        raise ValueError("do you want to -use_spark ?")
    if args.mnemonic == args.replay_mnemonic:
        raise ValueError("Same mnemonic is not allowed, to avoid confusion")
    if args.use_scalable_storage:
        logging.warning('Usage of -use_scalable_storage is deprecated as it is always true by default. '
                        'Use -no_scalable_storage if you need')

    args.strategy_args = convert_args_list_to_dict(args.strategy_args)

    # Options for the backtest runner. Contains both options for the Runner, and for the strategy framework
    backtest_runner_args = {
        'use_fixture_cache': args.cache,
        'save_n_days': args.save_n_days,
        # runs save_n_days+days_after+days_before at a time, than discard days after and before
        'days_after': args.days_post,  # at least one
        'days_before': args.days_prior,  # 0 only if you do not need any pre-match data
        'hours_after': args.hours_post,  # at least one
        'hours_before': args.hours_prior,  # 0 only if you do not need any pre-match data
        'use_spark': args.use_spark,
        'spark_driver_memory': '8g',
        'spark_executor_memory': '8g',
        'launcher_environment': args.launcher_environment,
        'spark_app_name': "backt2 {} {}_{}".format(args.strategy_name, args.start_date, args.end_date),
        'devpi_user': args.u,
        'devpi_index': args.i,
        'use_multiprocess': args.use_multiprocess,
        'max_spark_cores': args.max_spark_cores,
        'n_workers': args.n_workers,
        'store_instructions': not args.no_instructions,
        'last_state_only': args.last_state,
        'use_scalable_storage': not args.no_scalable_storage,
        'backtest_db': args.backtest_db,
        'results_coll': args.results_coll,
        'delete_only': args.delete_only,
        'store_signals': args.store_signals,
        'delay_bets_ms': args.delay_bets_ms,
        'odds_expiration_ms': args.odds_expiration_ms,
        'use_trades_matcher': args.use_trades_matcher,
    }

    if args.repopulate:
        backtest_runner_args.update({'repopulate': True})

    if args.capital is not None:
        backtest_runner_args.update({'allocated_capital': args.capital})

    return args, backtest_runner_args
