import Queue

from sgmtradingcore.strategies.dummy.strategy import DummyStrategy
from sgmtradingcore.util.spark_api import create_remote_context
from sgmtradingcore.backtesting.persistence import ensure_configurations
import datetime
import random
from threading import Thread
import numpy as np
import pytz
from sgmtradingcore.analytics.performance import dict_from_bet_state
from sgmtradingcore.analytics.metrics import flat_capital_metrics
from sgmtradingcore.backtesting.backtest import run_backtest
import collections
import pandas as pd
from sgmtradingcore.backtesting.persistence import BacktestOptimizationRun, insert_optimization_runs, BacktestConfig
from sgmtradingcore.backtesting.recorders import BackTestMongoOrderRecorder
from sgmtradingcore.backtesting.scoring import Scoring
from sgmtradingcore.core.enums import GsmCompetitions
from sgmtradingcore.core.trading_types import OrderStatus

import copy
from stratagemdataprocessing.dbutils.mongo import  MongoPersister


BACK = True
LAY = False

SELL = 'sell'
BUY = 'buy'
DONT_SELL_OR_BUY = 'dont_sell_or_buy'

STANDARD = 'standard'
WEIGHTED = 'weighted'

COVER_RISK = 'risk'
COVER_STAKE = 'stake'

CLOSE_ALWAYS = 'close_always'   # Closes positions on closing minute no matter what current score is
CLOSE_ON_DRAW = 'close_on_draw' # Closes positions only when current score is draw
CLOSE_ON_ZERO_DRAW = 'close_on_zero_draw' # Closes positions only when current score is 0-0
CLOSE_NEVER = 'close_never' # Never closes positions

FIXTURE_OPEN = 'open-positions'
FIXTURE_UNWIND = 'unwind-positions'
FIXTURE_CLOSE = 'close-positions'

MANDATORY_PARAMS = ['competitions', 'cover_type', 'closing_style', 'closing_time']


ALL_COMPS = [GsmCompetitions.PremierLeague,
             GsmCompetitions.Bundesliga1,
             GsmCompetitions.PrimeraDivision,
             GsmCompetitions.FranceLigue1,
             GsmCompetitions.ItalySerieA,
             GsmCompetitions.NetherlandsED,
             GsmCompetitions.AustriaBL,
             # GsmCompetitions.SwissSuperLeague,
             GsmCompetitions.ScottishPrem,
             GsmCompetitions.Bundesliga2,
             GsmCompetitions.PortugalPrimLiga,
             GsmCompetitions.EnglishChampionship]
TRAIN_COMPS = [GsmCompetitions.PremierLeague,
               GsmCompetitions.PrimeraDivision,
               GsmCompetitions.ItalySerieA,
               GsmCompetitions.AustriaBL,
               GsmCompetitions.Bundesliga2,
               GsmCompetitions.PortugalPrimLiga]
TEST_COMPS = [GsmCompetitions.Bundesliga1,
              GsmCompetitions.FranceLigue1,
              GsmCompetitions.NetherlandsED,
              # GsmCompetitions.SwissSuperLeague,
              GsmCompetitions.ScottishPrem,
              GsmCompetitions.EnglishChampionship]
SUMMER_COMPS = [GsmCompetitions.Tippeligaen,
                GsmCompetitions.Allsvenskan,
                GsmCompetitions.JapanJ1League,
                GsmCompetitions.USAMLS]


class PsoSpark:
    def __init__(self, func, lb, ub, ieqcons=[], f_ieqcons=None, not_continuos=False, args=(), kwargs={},
                 swarmsize=10, omega=0.5, phip=0.8, phig=0.2, maxiter=20,
                 minstep=1e-8, minfunc=1e-8, debug=False, number_of_cores=30, use_spark=False, context_params=None):
        raise ValueError("This algorithm is bound to the spread football strategy so until this gets dis entagled, it \
                         should not be used")
        self._lb = lb
        self._ub = ub
        self._f_ieqcons = f_ieqcons
        self._args = args
        self._kwargs = kwargs
        self._swarmsize = swarmsize
        self._omega = omega
        self._phip = phip
        self._phig = phig
        self._maxiter = maxiter
        self._minstep = minstep
        self._minfunc = minfunc
        self._debug = debug
        self._func = func
        self._cons = None
        self._ieqcons = ieqcons
        self._it = 0
        self._x = None  # Current position for each particle (continuos params)
        self._x_nc = None  # Current position for each particle (not continuos params)
        self._v = None  # Current velocity for each particle
        self._fg = 1e100  # artificial best swarm position starting value
        self._fp = None  # Best function value for each particle
        self._p = None  # Best position for each particle
        self._g = None  # Best solution ( domain )
        self._g_nc = None  # Best global position (not continuos params)
        self._p_nc = None  # Best position for each particle (not continuos params)
        self._fx = None  # Best solution ( codomain )
        self._obj = None
        self._parallelism = number_of_cores * 2
        self._not_continuos_features = not_continuos
        self._use_spark = use_spark
        self._context_params = context_params if context_params is not None else {}
        self._check_value_and_init(args, kwargs)

    def _check_value_and_init(self, args, kwargs):

        assert len(self._lb) == len(self._ub), 'Lower- and upper-bounds must be the same length'
        # assert hasattr(self._func, '__call__'), 'Invalid function handle'
        self._lb = np.array(self._lb)
        self._ub = np.array(self._ub)
        assert np.all(self._ub > self._lb), 'All upper-bound values must be greater than lower-bound values'
        vhigh = np.abs(self._ub - self._lb)
        vlow = - vhigh

        # Check for constraint function(s) #########################################
        if self._func:
            self._obj = lambda x: self._func(x, *args, **kwargs)
        if self._f_ieqcons is None:
            if not len(self._ieqcons):
                if self._debug:
                    print('No constraints given.')
                cons = lambda x: np.array([0])
            else:
                if self._debug:
                    print('Converting ieqcons to a single constraint function')
                cons = lambda x: np.array([y(x, *args, **kwargs) for y in self._ieqcons])
        else:
            if self._debug:
                print('Single constraint function given in f_ieqcons')
            cons = lambda x: np.array(self._f_ieqcons(x, *args, **kwargs))
        self._cons = cons

        self._terminate_init(vlow, vhigh)

    def _is_feasible(self, x):
        check = np.all(self._cons(x) >= 0)
        return check

    def process_result(self, res_):
        # Processing the result and if possible updating the local best value (particle best solution)
        # and best global result (best result across the whole swarm)
        for res in res_:

            fx_ = res.new_fx  # New function value
            x_ = res.new_x  # New position
            v_ = res.new_v  # New velocity
            i_ = res.i  # Number of the item inside the population
            if self._not_continuos_features:
                x_nc_ = res.x_nc
                self._x_nc[i_] = x_nc_[i_]
            # Update position and velocity
            self._v[i_, :] = v_[i_, :]
            self._x[i_, :] = x_[i_, :]

            # Checking if a new minimum has been hit and in case updating best position and function value for
            # this particular item (later there is the check for global minimum)

            if fx_ < self._fp[i_] and self._is_feasible(x_[i_, :]):
                self._p[i_, :] = x_[i_, :].copy()
                if self._not_continuos_features:
                    self._p_nc[i_] = x_nc_[i_]
                self._fp[i_] = fx_

                # Compare swarm's best position to current particle's position
                # (Can only get here if constraints are satisfied)
                if fx_ < self._fg:
                    tmp = x_[i_, :].copy()
                    if self._not_continuos_features:
                        tmp_nc = copy.deepcopy(x_nc_[i_])
                    stepsize = np.sqrt(np.sum((self._g - tmp) ** 2))
                    if np.abs(self._fg - fx_) <= self._minfunc:
                        print(
                            'Stopping search: Swarm best objective change less than {:}'.format(self._minfunc))
                        self._g = tmp.copy()
                        if self._not_continuos_features:
                            self._g_nc = copy.deepcopy(tmp_nc)
                        self._fg = fx_
                        return 1
                    elif stepsize <= self._minstep:
                        print('Stopping search: Swarm best position change less than {:}'.format(self._minstep))
                        self._g = tmp.copy()
                        if self._not_continuos_features:
                            self._g_nc = tmp_nc
                        self._fg = fx_
                        return 1
                    else:
                        self._g = tmp.copy()
                        if self._not_continuos_features:
                            self._g_nc = copy.deepcopy(tmp_nc)
                        self._fg = fx_
        return 0

    def _terminate_init(self, vlow, vhigh):

        # Initialize the particle swarm ############################################

        self._S = self._swarmsize  # Number of particles
        self._D = len(self._lb)  # the number of dimensions of each particle
        self._x = np.random.rand(self._S, self._D)  # particle positions
        self._v = np.zeros_like(self._x)  # particle velocities
        self._p = np.zeros_like(self._x)  # best particle positions
        self._fp = np.zeros(self._S)  # best particle function values
        self._g = []  # best swarm position

        if not self._not_continuos_features:
            for i in range(self._S):
                # Initialize the particle's position
                self._x[i, :] = self._lb + self._x[i, :] * (self._ub - self._lb)
                # Initialize the particle's best known position
                self._p[i, :] = self._x[i, :]
                # Initialize the particle's velocity
                self._v[i, :] = vlow + np.random.rand(self._D) * (vhigh - vlow)

        # Here is the case for NON continuos features
        else:

            self._x_nc = [[random.random() for e in range(len(self._values_for_each_not_continuos_param))] for e in
                          range(self._S)]
            self._p_nc = [[0 for e in range(len(self._values_for_each_not_continuos_param))] for e in range(self._S)]

            for i in range(self._S):
                index_nc = 0  # Index for categorical features
                for index in range(len(self._v[1])):
                    if index not in self._values_for_each_not_continuos_param:
                        self._x[i, index] = self._lb[index] + self._x[i, index] * (self._ub[index] - self._lb[index])
                        # Initialize the particle's best known position
                        self._p[i, index] = self._x[i, index]
                        # Initialize the particle's velocity
                        self._v[i, index] = vlow[index] + np.random.rand(self._D)[index] * (vhigh[index] - vlow[index])
                    else:
                        # get all values for that feature
                        values = self._values_for_each_not_continuos_param[index]
                        # choose one index randomly
                        index_to_set = random.randint(0, len(values) - 1)
                        value_to_set = values[index_to_set]
                        # Update position for this feature and this particle
                        self._x_nc[i][index_nc] = value_to_set
                        self._p_nc[i][index_nc] = self._x_nc[i][index_nc]
                        index_nc += 1

        if self._use_spark:
            sc = create_remote_context(num_cores=30, **self._context_params)

        def init(i):
            # Calculate the objective's value at the current particle's
            if self._obj:
                return self._obj(self._p[i, :]), i
            else:
                return self._run_fun(self._p, self._p_nc, i, initial_=True), i

        # need to update best solution for each particle given the above initialization
        b = 0
        e = 0

        while e < self._S:
            e += self._parallelism
            if e > self._S:
                e = self._S
            l = range(b, e)

            if self._use_spark:
                paral = min(self._parallelism, len(l))
                rdd = sc.parallelize(l, paral)
                res = rdd.map(init).collect()
            else:
                res = map(lambda x: init(x), l)

            b = e
            # Iterate on results and store solutions for each particle
            for j in res:

                r = j[0]  # result
                i = j[1]  # particle index
                self._fp[i] = r

                # At the start, there may not be any feasible starting point, so just
                # give it a temporary "best" point since it's likely to change
                if i == 0:
                    self._g = self._p[0, :].copy()
                    if self._not_continuos_features:
                        self._g_nc = copy.deepcopy(self._p_nc[0])

                # If the current particle's position is better than the swarm's,
                # update the best swarm position
                if self._fp[i] < self._fg and self._is_feasible(self._p[i, :]):
                    self._fg = self._fp[i]
                    self._g = self._p[i, :].copy()
                    if self._not_continuos_features:
                        # Store categorical input
                        self._g_nc = copy.deepcopy(self._p_nc[i])

        # In case save spark context for reusing that as soon as evaluate is called
        # 2 spark context are not allowed and current implementation does not follow singleton pattern
        if self._use_spark:
            self._sc = sc

    def evolve(self):
        it = 1
        # Save locally the spark context and cancel reference in self, because it is not pickable
        if self._use_spark:
            sc = self._sc
            self._sc = None

        particles = range(self._S)
        executing = {}
        num_execution = {}
        number_of_cores = self._parallelism
        level_of_parallelism = self._parallelism
        queue_to_execute = Queue.Queue(maxsize=level_of_parallelism)
        queue_done = Queue.Queue(maxsize=level_of_parallelism)

        # Thread function: it runs a spark job and send results to the main thread using Queue
        def worker():
            while True:
                param, particle = queue_to_execute.get()
                num_part = min(number_of_cores, len(param))
                if self._use_spark:
                    rdd = sc.parallelize(param, num_part)
                    result = rdd.map(self._update).collect()
                else:
                    result = map(lambda x: self._update(x), param)
                queue_done.put((result, particle))
                queue_to_execute.task_done()

        for i in particles:
            executing[i] = False
            num_execution[i] = 0

        for _ in range(level_of_parallelism):
            t = Thread(target=worker)
            t.daemon = True
            t.start()

        working = 0

        # Iterate until termination criterion met ###
        while it <= self._maxiter:

            print 'starting iteration: ' + str(it)
            rp = np.random.uniform(size=(self._S, self._D))  # Cognitive coeff
            rg = np.random.uniform(size=(self._S, self._D))  # Social coeff

            for i in particles:
                self._it = it
                param = []
                Tuple = collections.namedtuple("Param",
                                               ['v', 'rp', 'p', 'x', 'i', 'rg', 'g', 'lb', 'ub', 'x_nc'])
                param.append(Tuple(self._v, rp, self._p, self._x, i, rg, self._g, self._lb, self._ub, self._x_nc))

                # Do not evaluate a particle that is being currently evaluated
                if executing[i]:
                    continue

                # Insert into the 'ready to run' queue
                queue_to_execute.put((param, i))

                # Setting this particular particle as 'running'
                executing[i] = True

                # Increment number of particles that are being evaluated
                working += 1

                # Increment the number of evaluations
                it += 1

                # Upper-bound about the number of particles that can be evaluating in the same time
                if working < level_of_parallelism and working < self._S and it < self._maxiter:
                    continue

                # Asynchronous call: get the results from the first particle that has finished to be evaluated
                res_, particle_index = queue_done.get()

                # Increment the number of executions for this particle
                num_execution[particle_index] += 1

                # Setting as ready to be evaluated again
                executing[particle_index] = False

                # Free a slot
                working -= 1

                stop = self.process_result(res_)

                # Check number of iterations
                if it > self._maxiter or stop == 1:
                    it = self._maxiter + 1
                    break

        # Wait until all remaining threads are completed
        while working > 0:
            res_, _ = queue_done.get()
            working -= 1
            self.process_result(res_)

        if not self._is_feasible(self._g):
            print("However, the optimization couldn't find a feasible design. Sorry")

        return

    def get_result(self):
        return self._g, self._fg

    def _run_fun(self, x, i):
        return self._obj(x[i, :])

    def _update(self, param):

        v = param.v  # Particle velocity
        rp = param.rp
        p = param.p  # Particle best position, so far
        x = param.x  # Particle current position
        i = param.i  # Particle Index
        rg = param.rg
        g = param.g  # Best solution so far
        lb = param.lb  # lower bound
        ub = param.ub  # upper bound
        if self._not_continuos_features:
            x_nc = param.x_nc

        if not self._not_continuos_features:
            # Update the particle's velocity
            v[i, :] = self._omega * v[i, :] + self._phip * rp[i, :] * (p[i, :] - x[i, :]) + \
                      self._phig * rg[i, :] * (g - x[i, :])
            x[i, :] = x[i, :] + v[i, :]
        else:
            index_nc = 0
            for index in range(len(v[1])):
                if index not in self._values_for_each_not_continuos_param:
                    v[i, index] = self._omega * v[i, index] + self._phip * rp[i, index] * (p[i, index] - x[i, index]) + \
                                  self._phig * rg[i, index] * (g[index] - x[i, index])
                    x[i, index] = x[i, index] + v[i, index]
                else:
                    list_to_extract_from = []
                    best_value_local = self._p_nc[i][index_nc]
                    best_value_global = self._g_nc[index_nc]
                    values = self._values_for_each_not_continuos_param[index]
                    prob_local = 0.25
                    prob_global = 0.35
                    if len(values) == 2:
                        if best_value_global != best_value_local:
                            prob_local = 0.45
                            prob_global = 0.55
                            others_prob = 0
                        else:
                            prob_global = 0.45
                            prob_local = 0.35
                            others_prob = 0.2
                    else:
                        others_prob = (1 - (prob_local + prob_global)) / (len(values) - 2)
                    prob_local = int(prob_local * 100.0)
                    prob_global = int(prob_global * 100.0)
                    others_prob = int(others_prob * 100.0)

                    for f in values:
                        if f == best_value_local:
                            list_to_extract_from += [f] * prob_local
                        if f == best_value_global:
                            list_to_extract_from += [f] * prob_global
                        if f != best_value_local and f != best_value_global:
                            list_to_extract_from += [f] * others_prob

                    value_to_set = random.choice(list_to_extract_from)
                    x_nc[i][index_nc] = value_to_set
                    index_nc += 1

        # Update the particle's position, correcting lower and upper bound
        # violations, then update the objective function value

        mark1 = x[i, :] < lb
        mark2 = x[i, :] > ub
        x[i, mark1] = lb[mark1]
        x[i, mark2] = ub[mark2]
        if not self._not_continuos_features:
            fx = self._run_fun(x, i)
            Return = collections.namedtuple("Return", ['new_fx', 'new_x', 'new_v', 'i'])
            return Return(fx, x, v, i)
        else:
            fx = self._run_fun(x, x_nc, i)
            Return = collections.namedtuple("Return", ['new_fx', 'new_x', 'new_v', 'i', 'x_nc'])
            return Return(fx, x, v, i, x_nc)


class StrategyChooseParamsPSO(PsoSpark):
    def __init__(self, optimization_name, start_time, end_time, not_continuos, ieqcons=[],
                 f_ieqcons=None, args=(), kwargs={}, swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=450,
                 minstep=1e-8, minfunc=1e-8, debug=False, use_spark=False, context_params=None):

        self._start_time = start_time
        self._end_time = end_time
        self._values_for_each_not_continuos_param = {}
        self._score_cache = {}
        self._optimization_name = optimization_name
        lb, ub = self._create_boundaries()
        PsoSpark.__init__(self, None, lb, ub, ieqcons, f_ieqcons, not_continuos, args, kwargs,
                          swarmsize, omega, phip, phig, maxiter,
                          minstep, minfunc, debug, use_spark=use_spark, context_params=context_params)

    def _create_boundaries(self):
        raise NotImplementedError('Should be implemented by subclass')

    @staticmethod
    def strategy_name():
        """
        The globally unique human readable strategy name
        """
        raise NotImplementedError('Should be implemented by subclass')

    def strategy_instance(self, ind):
        raise NotImplementedError('Should be implemented by subclass')

    @staticmethod
    def opt_static_params():
        """
        Return the static parameters, ie which will not be part of the optimization
        """
        raise NotImplementedError('Should be implemented by subclass')


    @staticmethod
    def get_capital_by_day_as_df(backtest_output):

        risk_dict = {}
        last_day = None
        for time, capital_obj in backtest_output['risk_provider'][backtest_output['risk_provider'].keys()[0]]. \
                iteritems():
            current_day = time.date()
            if current_day != last_day:
                capital = capital_obj.capital
                risk_dict[current_day] = capital
                last_day = current_day
        risk_provider_pd = pd.DataFrame(risk_dict.items(), columns=['date', 'capital'])
        return risk_provider_pd

    def save_on_backtest_results(self, conn, strategy_name, dynamic_param, output):

        recorder = BackTestMongoOrderRecorder(conn, strategy_name,
                                              self._optimization_name)
        recorder.record(
            self._start_time, self._end_time, self.static_params(), dynamic_param, output,
            datetime.datetime.now(pytz.UTC))

    def save_on_backtest_optimization(self, conn, dynamic_params, score, metrics, optimization_params):
        runs = []
        strategy_config_id = BacktestConfig(FootballSpreadChooseParamsPSO.strategy_name(), self.static_params(),
                                            dynamic_params)
        ensure_configurations(conn, FootballSpreadChooseParamsPSO.strategy_name(), [strategy_config_id],
                              datetime.datetime.now(pytz.UTC))
        strategy_config_id = strategy_config_id.object_id
        runs.append(
            BacktestOptimizationRun(strategy_config_id, self._start_time, self._end_time, -score, metrics,
                                    optimization_params
                                    )
        )
        insert_optimization_runs(conn, self._optimization_name, runs, datetime.datetime.now(pytz.UTC))

    def get_score(self, backtest_output):
        """
        This is the score used by the ALGO, defaults to _calculate_score function but override as necessary.
        The simple case is to override _calculate_score to use a different score from the
        metrics or override this method to do something completely different.
        """
        if len(backtest_output['bet_states']):
            bets_df = pd.DataFrame([
                                       dict_from_bet_state(dt_, bs)
                                       for dt_, bs in backtest_output['bet_states'].iteritems()
                                       if bs.status == OrderStatus.SETTLED and bs.matched_odds > 1.
                                       ]
                                   )
            risk_provider_pd = StrategyChooseParamsPSO.get_capital_by_day_as_df(backtest_output)
            metrics = flat_capital_metrics(bets_df, capital=risk_provider_pd)
            if metrics is None:
                return float('inf')
            else:
                metrics = metrics.iloc[0]

            strategy_descrs = backtest_output['risk_provider'].keys()
            if len(strategy_descrs) > 1:
                raise ValueError('Optimizer does not support multiple strategy descrs')

            strategy_descr = strategy_descrs[0]
            incr_metrics = backtest_output['risk_provider'][strategy_descr].iloc[-1]

            metrics_dict = {
                'drawdown': metrics['maximum_drawdown'],
                'cum_return': metrics['cum_return'],
                'total_pnl': metrics['total_pnl'],
                'n_win': metrics['n_win'],
                'n_loss': metrics['n_loss'],
                'volatility': metrics['volatility (not annualised)'],
                'time_in_market': incr_metrics.time_in_market,
                'rina': incr_metrics.rina
            }

            return self._calculate_score(metrics), metrics_dict
        else:
            return float('inf'), None

    def _calculate_score(self, metrics):
        """
        Override to calculate the score using one or more of the standard metrics
        """
        return -Scoring.calculate_general_score(metrics)
        # return -Scoring.calculate_score_100(metrics)

class FootballSpreadChooseParamsPSO(StrategyChooseParamsPSO):
    def __init__(self, opt_name, start_time, end_time, swarm_size=100, max_iter=450, use_spark=False,
                 context_params=None):

        StrategyChooseParamsPSO.__init__(self, opt_name, start_time, end_time,
                                         True, swarmsize=swarm_size, maxiter=max_iter, use_spark=use_spark,
                                         context_params=context_params)
        self._param_def = None

    @staticmethod
    def strategy_name():
        return 'butterfly'

    def strategy_instance(self, ind):
        strategy_params = self.static_params()
        strategy_params.update(ind)
        # return FootballSpreadStrategy(**strategy_params)
        return DummyStrategy()

    def _create_boundaries(self):
        lb = []
        ub = []

        for i, (k, v) in enumerate(self.param_def.iteritems()):
            if v['type'] != 'continuous':
                self._not_continuos_features = True
                self._values_for_each_not_continuos_param[i] = v['args']
                lb.append(0)
                ub.append(1)
            else:
                lb.append(v['args'][0])
                ub.append(v['args'][1])
        return lb, ub

    def _run_fun(self, x, x_nc, i, initial_=False):

        params_to_use = self.static_params()
        dinamic_params = {}
        p = self.param_def
        i_nc = 0

        for index, (k, v) in enumerate(p.iteritems()):
            if index not in self._values_for_each_not_continuos_param:
                dinamic_params[k] = x[i, index]
            else:
                dinamic_params[k] = x_nc[i][i_nc]
                i_nc += 1

        params_to_use.update(dinamic_params)
        params_to_use = self.modify_parameters(params_to_use, initial=initial_)
        for k in dinamic_params:
            dinamic_params[k] = params_to_use[k]
        strategy_instance = self.strategy_instance(params_to_use)
        output = run_backtest(strategy_instance, self._start_time, self._end_time)
        conn = MongoPersister.init_from_config('trading_dev', auto_connect=True)
        self.save_on_backtest_results(conn, FootballSpreadChooseParamsPSO.strategy_name(), dinamic_params, output)
        score, metrics = self.get_score(output)
        optimization_params = {'particle': i, 'iteration': self._it}
        self.save_on_backtest_optimization(conn, dinamic_params, score, metrics, optimization_params)
        return score

    def get_result(self):
        params_to_return = {}
        p = self.param_def
        index_nc = 0
        for index, (k, v) in enumerate(p.iteritems()):
            if index not in self._values_for_each_not_continuos_param:
                params_to_return[k] = self._g[index]
            else:
                params_to_return[k] = self._g_nc[index_nc]
                index_nc += 1
        return params_to_return, self._fg

    def static_params(self):
        return {'competitions': TRAIN_COMPS,
                'limit_supremacy': None,
                'use_empirical': False,
                'hedge_style': WEIGHTED,
                'risk_on_the_draw': 1000
                }

    @staticmethod
    def modify_parameters(ind, initial=False):
        # Ensure that weights don't add to more than 1
        w1, w2 = ind['weight_param_1'], ind['weight_param_2']
        if initial:
            if w1 + w2 > 1.:
                # Reflect so that random weights are uniform in {x + y < 1 | x,y in [0,1]x[0,1]}
                ind['weight_param_1'], ind['weight_param_2'] = 1. - w2, 1. - w1
        else:
            if w1 + w2 > 1.:
                # Snap back to the line as algorithm is trying to extend beyond there
                ind['weight_param_1'] /= w1 + w2
                ind['weight_param_2'] /= w1 + w2

        # If close never is chosen then modify closing time
        if ind['closing_style'] == CLOSE_NEVER:
            ind['closing_time'] = 100
        elif ind['closing_time'] == 100:
            ind['closing_style'] = CLOSE_NEVER

        return ind

    @staticmethod
    def opt_params(short=False):
        """
        The genetic algorithm parameters
        """
        if short:
            return {'swarm_size': 100, 'max_iter': 500}
        else:
            return {'swarm_size': 150, 'max_iter': 750}

    @staticmethod
    def date_ranges():
        """
        The date ranges to cross validate for
        """
        ranges = [
            (datetime.datetime(2015, 8, 1, tzinfo=pytz.UTC), datetime.datetime(2015, 12, 31, tzinfo=pytz.UTC)),
        ]

        return ranges

    @property
    def param_def(self):
        return collections.OrderedDict([('weight_param_1', {'type': 'continuous', 'args': [0, 1]}),
                                        ('weight_param_2', {'type': 'continuous', 'args': [0, 1]}),
                                        ('closing_time', {'type': 'ordinal', 'args': [10, 20, 30, 40, 50, 60, 100]}),
                                        ('closing_style', {'type': 'ordinal', 'args': [CLOSE_ALWAYS, CLOSE_ON_DRAW,
                                                                                       CLOSE_ON_ZERO_DRAW,
                                                                                       CLOSE_NEVER]}),
                                        ('cover_type', {'type': 'categorical', 'args': [COVER_RISK, COVER_STAKE]}),
                                        ('indicator_threshold', {'type': 'continuous', 'args': [0.01, 0.20]}),
                                        ('fixture_supremacy_threshold', {'type': 'continuous', 'args': [-1.0, 1.0]}),
                                        ('fixture_total_goals_threshold', {'type': 'continuous', 'args': [2.5, 3.0]}),
                                        ('fixture_bl_scale', {'type': 'continuous', 'args': [0.1, 0.9]}),
                                        ('fixture_br_scale', {'type': 'continuous', 'args': [0.1, 0.9]}),
                                        ('fixture_tl_scale', {'type': 'continuous', 'args': [0.1, 0.9]}),
                                        ('fixture_tr_scale', {'type': 'continuous', 'args': [0.1, 0.9]})])
