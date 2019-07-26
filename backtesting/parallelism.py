import logging
import timeit
from datetime import datetime
import pytz

from sgmtradingcore.backtesting.persistence import BacktestConfig, ensure_configurations, BacktestOptimizationRun, \
    insert_optimization_runs, generate_config_key
from sgmtradingcore.util.spark_api import create_remote_context
from stratagemdataprocessing.dbutils.mongo import MongoPersister
from sgmtradingcore.backtesting.common import PARAM_CLASS_MAP


class MappingFunctions(object):
    """
    Utility class for running the backtest locally
    """
    def __init__(self, strategy_name, optimization_name, static_params, start_time, end_time, conn):
        self._start_time = start_time
        self._end_time = end_time
        self._strategy_name = strategy_name
        self._optimization_name = optimization_name
        self._static_params = static_params
        self._conn = conn

        self._score_cache = {}
        self._num_calls = 0

    def _get_config_ids_from_parameters(self, param_list):
        configs = []

        dt = datetime.now(pytz.UTC)
        for ind in param_list:
            static_params = self._static_params
            dynamic_params = dict(ind)
            configs.append(BacktestConfig(self._strategy_name, static_params, dynamic_params))

        ensure_configurations(self._conn, self._strategy_name, configs, dt)
        return [config.object_id for config in configs]

    def _do_map(self, function, param_list_to_calc):
        res = map(function, param_list_to_calc)

        return res

    def map(self, function, param_list):
        param_list_to_calc = []

        for ind in param_list:
            key = generate_config_key(ind)

            if key not in self._score_cache:
                param_list_to_calc.append(dict(ind))

        res = self._do_map(function, param_list_to_calc)

        for result, ind in zip(res, param_list_to_calc):
            key = generate_config_key(ind)
            score, metrics = result

            self._score_cache[key] = (score, metrics)

        scores = list()
        runs = []
        strategy_config_ids = self._get_config_ids_from_parameters(param_list)

        optimization_params = {'generation': self._num_calls}

        for strategy_config_id, current_param in zip(strategy_config_ids, param_list):
            key = generate_config_key(current_param)
            score, metrics = self._score_cache[key]

            runs.append(
                BacktestOptimizationRun(
                    strategy_config_id, self._start_time, self._end_time, score, metrics, optimization_params
                )
            )
            scores.append((score,))

        insert_optimization_runs(self._conn, self._optimization_name, runs, datetime.now(pytz.UTC))
        self._num_calls += 1

        return scores


class EvaluatorClass(object):
    """
    This object gets streamed across the spark cluster from the driver
    so needs to be pickleable
    """

    def __init__(self, strategy, optimization, start_time, end_time, static_params):
        self._strategy = strategy
        self._opt_name = optimization
        self._start_time = start_time
        self._end_time = end_time
        self._static_params = static_params

    def evaluate_spark(self, ind):
        # Creating connection here in the worker
        conn = MongoPersister.init_from_config('trading_dev', auto_connect=True)
        strategy_GA = PARAM_CLASS_MAP[self._strategy](self._opt_name, conn, self._start_time, self._end_time,
                                                      self._static_params)

        return strategy_GA.evaluate_spark(ind)


class SparkFunctions(MappingFunctions):
    def __init__(self, strategy_name, optimization_name, static_params, start_time, end_time, conn,
                 context_params):
        MappingFunctions.__init__(self, strategy_name, optimization_name, static_params, start_time, end_time, conn)

        self._spark_context = create_remote_context(**context_params)

    def _do_map(self, function, param_list_to_calc):
        logging.info('Evaluating %d individuals using Spark' % len(param_list_to_calc))

        eval_ = EvaluatorClass(self._strategy_name, self._optimization_name,
                               self._start_time, self._end_time, self._static_params)

        start_time = timeit.default_timer()
        # From here: http://spark.apache.org/docs/latest/tuning.html
        # " In general, we recommend 2-3 tasks per CPU core in your cluster. "
        # Math0x has 12 cores each one, so:
        number_of_nodes = 2
        number_partition = 12 * number_of_nodes * 2
        rdd_param = self._spark_context.parallelize(param_list_to_calc, number_partition)
        results_from_spark = rdd_param.map(eval_.evaluate_spark).collect()

        elapsed = timeit.default_timer() - start_time
        elapsed_min = int(elapsed / 60)
        logging.info('Evaluated %d individuals in %d minutes' % (len(param_list_to_calc), elapsed_min))

        return results_from_spark


class LocalFunctions(MappingFunctions):

    def _do_map(self, function, param_list_to_calc):
        logging.info('Evaluating %d individuals' % len(param_list_to_calc))

        eval_ = EvaluatorClass(self._strategy_name, self._optimization_name, self._start_time, self._end_time,
                               self._static_params)

        start_time = timeit.default_timer()
        results = map(eval_.evaluate_spark, param_list_to_calc)
        elapsed = timeit.default_timer() - start_time
        elapsed_min = int(elapsed / 60)
        logging.info('Evaluated %d individuals in %d minutes' % (len(param_list_to_calc), elapsed_min))

        return results
