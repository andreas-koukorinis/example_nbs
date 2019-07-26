import time
from datetime import timedelta

from sgmtradingcore.core.ts_log import get_logger
from sgmtradingcore.providers.odds_providers import HistoricalFileOddsProvider
from sgmtradingcore.backtesting.backtest import run_backtest
from sgmtradingcore.backtesting.recorders import BackTestMetricsCSVRecorder
from stratagemdataprocessing.bookmakers.common.odds.cache import HistoricalOddsCache


def run_looping_backtest(strategy_class, parameters, start_date, end_date, result_csv_file,
                         skip_loops=0, aux_recorder=None, model_provider=None, trade_days_per_year=330):
    logger = get_logger(__name__)
    loop_no = 0
    loop_times = []
    odds_cache = HistoricalOddsCache()

    param_columns = list(parameters[0].iterkeys())
    param_columns.sort()

    metrics_recorder = BackTestMetricsCSVRecorder(result_csv_file, param_columns, trade_days_per_year=trade_days_per_year)

    for param in parameters:
        loop_no += 1

        if loop_no <= skip_loops:
            logger.info(None, 'Skipping backtest loop #%d' % (loop_no,))
            continue

        if len(loop_times) > 0:
            remaining_loops = len(parameters) - loop_no + 1
            mean_loop_time = sum(loop_times) / len(loop_times)
            remaining_time = timedelta(seconds=mean_loop_time * remaining_loops)
            logger.info(None, 'Running backtest loop #%d of %d. Estimated time remaining: %s' % (loop_no, len(parameters), remaining_time))
        else:
            logger.info(None, 'Running backtest loop #%d of %d. Estimated time remaining: N/A' % (loop_no, len(parameters)))

        loop_start = time.time()
        strategy = strategy_class(**param)

        strategy_output = run_backtest(
            strategy, start_date, end_date,
            odds_provider=HistoricalFileOddsProvider(odds_cache=odds_cache),
        )

        loop_time = time.time() - loop_start
        loop_times.append(loop_time)
        metrics_recorder.record(loop_no, loop_time, start_date, end_date, param, strategy_output)
        if aux_recorder is not None:
            aux_recorder.record(loop_no, loop_time, start_date, end_date, param, strategy_output)
