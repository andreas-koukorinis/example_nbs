import csv
import logging
import os.path
import sys
from datetime import datetime

import pytz

from sgmtradingcore.analytics.performance import convert_bet_states_to_array
from sgmtradingcore.backtesting.persistence import (
    BacktestConfig, ensure_configurations, BacktestOrder, BacktestResults, save_run_results)
from sgmtradingcore.core.trading_types import OrderStatus


class BackTestCSVRecorderBase(object):
    def __init__(self, output_file, overwrite=False):
        if os.path.exists(output_file):
            if overwrite:
                os.unlink(output_file)
            else:
                raise ValueError('Report file %s already exists. Delete it before starting backtest.' % output_file)
        self._output_file = output_file

    def record(self, loop_no, loop_time, start_date, end_date, params, strategy_output):
        raise NotImplementedError('Please implement this method')

    def _save(self):
        # Save results to file after each loop
        data = [[unicode(val).encode('utf-8') if isinstance(val, basestring) else val
                 for val in row] for row in self._metrics]

        try:
            with open(self._output_file, 'wb') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(self._output_columns)
                csv_writer.writerows(data)
        except IOError:
            # Fallback to stdout to avoid losing results
            logging.exception('Writing backtest results to "%s" failed. Printing on stdout instead.' % self._output_file)
            csv_writer = csv.writer(sys.stdout)
            csv_writer.writerow(self._output_columns)
            csv_writer.writerows(data)


class BackTestMetricsCSVRecorder(BackTestCSVRecorderBase):
    def __init__(self, output_file, param_columns, capital=10000, trade_days_per_year=330):
        super(BackTestMetricsCSVRecorder, self).__init__(output_file)

        self._metrics = []
        self._param_columns = param_columns
        self._result_columns = [
            'CR_day', 'CR_trade', 'Pounds/Stake', 'annualised_return', 'avgTradeLosses', 'avgTradeWins',
            'calmar_ratio', 'drawdown', 'drawdown_pct', 'nTradeLosses', 'nTradeWins', 'runup', 'runup_pct',
            'sharpe_ratio','sortino_ratio', 'volatility', 'total_trades', 'total_stake', 'total_pnl']

        self._output_columns = ['loop', 'time', 'start', 'end']
        self._output_columns.extend(self._param_columns)
        self._output_columns.extend(self._result_columns)

        self._capital = capital
        self._trade_days_per_year = trade_days_per_year

    def record(self, loop_no, loop_time, start_date, end_date, params, strategy_output):
        loop_bets = convert_bet_states_to_array(strategy_output['bet_states'])
        raise NotImplementedError('Functions from analytics/metrics.py need to be integrated here')
        # loop_metrics = performance_metrics_net(
        #     loop_bets, capital=self._capital, trade_days_per_year=self._trade_days_per_year)
        loop_time_str = '%sm%ss' % (divmod(int(loop_time), 60))
        loop_result = [loop_no, loop_time_str, start_date, end_date]

        for key in self._param_columns:
            loop_result.append(params.get(key, 'N/A'))

        for key in self._result_columns:
            loop_result.append(loop_metrics.get(key, 'N/A'))

        self._metrics.append(loop_result)
        self._save()


class BackTestMongoRecorderBase(object):

    def __init__(self, output_file):
        if os.path.exists(output_file):
            raise ValueError('Report file %s already exists. Delete it before starting backtest.' % output_file)
        self._output_file = output_file

    def record(self, loop_no, loop_time, start_date, end_date, params, strategy_output):
        raise NotImplementedError('Please implement this method')


class BackTestMongoOrderRecorder(object):

    def __init__(self, conn, strategy_name, opt_name):
        self._conn = conn
        self._strategy_name = strategy_name
        self._opt_name = opt_name

    def record(self, start_date, end_date, static_params, variable_params, strategy_output, dt):
        config = BacktestConfig(self._strategy_name, static_params, variable_params)

        ensure_configurations(self._conn, self._strategy_name, [config], dt)

        backtest_orders = {}

        for timestamp, bet_info in strategy_output['bet_states'].iteritems():
            if bet_info.id not in backtest_orders:
                backtest_order = BacktestOrder(
                    bet_info.sticker, bet_info.is_back, bet_info.matched_odds,
                    bet_info.matched_amount, bet_info.price, bet_info.size, bet_info.details, timestamp)

                backtest_orders[bet_info.id] = backtest_order

            if bet_info.status == OrderStatus.SETTLED:
                backtest_orders[bet_info.id].odds = bet_info.matched_odds
                backtest_orders[bet_info.id].size = bet_info.matched_amount
                backtest_orders[bet_info.id].settled_dt = timestamp
                backtest_orders[bet_info.id].pnl = bet_info.pnl

        result = BacktestResults(self._opt_name, config.object_id, backtest_orders.values())

        save_run_results(self._conn, result, datetime.now(pytz.UTC))

        return result
