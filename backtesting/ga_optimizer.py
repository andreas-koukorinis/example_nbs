import logging
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import pytz
from itertools import product
from deap import base, creator, tools, algorithms

from sgmtradingcore.analytics.metrics import flat_capital_metrics
from sgmtradingcore.backtesting.backtest import run_backtest
from sgmtradingcore.backtesting.recorders import BackTestMongoOrderRecorder
from sgmtradingcore.backtesting.scoring import Scoring
from sgmtradingcore.core.trading_types import OrderStatus
from sgmtradingcore.analytics.performance import dict_from_bet_state
from sgmtradingcore.analytics.performance import apply_betfair_commission


class ChooseParamsGA(object):
    """
    This class provides functionality to optimise the parameters of a strategy using a genetic algorithm
    """

    def __init__(self, npop=50, nhof=1, indpb=0.5, tournsize=5, mu=50, lambda_=20, cxpb=0.5, mutpb=0.2, ngen=10,
                 gridsearch=False, cont_slices=5):
        creator.create('FitnessMax', base.Fitness, weights=self.weights)
        creator.create('Individual', dict, fitness=creator.FitnessMax)

        self._npop = npop  # Number of individuals in initial population
        self._nhof = nhof  # Number of individuals in hall of fame
        self._tournsize = tournsize
        self._indpb = indpb  # Probability of mutation
        self._mu = mu  # The number of individuals to select for the next generation.
        self._lambda = lambda_  # The number of children to produce at each generation.
        self._cxpb = cxpb  # The probability that an offspring is produced by crossover.
        self._mutpb = mutpb  # The probability that an offspring is produced by mutation
        self._ngen = ngen  # The number of generation.

        self._gridsearch = gridsearch
        if gridsearch:
            # Mode to perform a full grid search in the first generation
            if npop is not None:
                raise ValueError('Population number cannot be specified in grid search mode')
            if mu is not None:
                raise ValueError('Next generation size (mu) cannot be specified in grid search mode')
            if lambda_ != 0:
                raise ValueError('Number of children (lambda_) should be 0 in grid search mode')
            if ngen != 1:
                raise ValueError('Number of generations should be 1 in grid search mode')
        self._cont_slices = cont_slices

        self._toolbox = None
        self._param_def = None
        self._hof = None
        self._stats = None
        self._pop = None
        self._logbook = None

    @property
    def weights(self):
        return 1.,

    @property
    def param_def(self):
        raise NotImplementedError('Should be implemented by subclass')

    def evaluate(self, ind):
        raise NotImplementedError('Should be implemented by subclass')

    @property
    def toolbox(self):
        if self._toolbox is None:
            toolbox = base.Toolbox()

            toolbox.register('attr_params', self.f_initial_individual)
            toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_params)
            toolbox.register('population', tools.initRepeat, list, toolbox.individual)
            toolbox.register('evaluate', self.evaluate)
            toolbox.register('mate', self.f_mate)
            toolbox.register('mutate', self.f_mutate, indpb=self._indpb)
            toolbox.register('select', tools.selTournament, tournsize=self._tournsize)
            self._toolbox = toolbox

        return self._toolbox

    @property
    def pop(self):
        if self._pop is None:
            if self._gridsearch:
                self._pop = self.expand_param_grid()
                self._mu = len(self._pop)  # Only 1 generation so carry all individuals through
            else:
                self._pop = self.toolbox.population(self._npop)
        return self._pop

    @pop.setter
    def pop(self, x):
        self._pop = x

    @property
    def hof(self):
        if self._hof is None:
            self._hof = tools.HallOfFame(self._nhof)
        return self._hof

    @hof.setter
    def hof(self, x):
        self._hof = x

    @property
    def logbook(self):
        if self._logbook is None:
            self._logbook = []
        return self._logbook

    @logbook.setter
    def logbook(self, x):
        self._logbook = self.logbook + x

    @staticmethod
    def modify_parameters(ind, initial=True):
        return ind

    @staticmethod
    def get_individual(ind):
        return creator.Individual(ind)

    @staticmethod
    def f_initial_bit_value(p):
        if p['type'] == 'continuous':
            return np.random.random() * (p['args'][1] - p['args'][0]) + p['args'][0]
        elif p['type'] in ['ordinal', 'categorical']:
            return np.random.choice(p['args'])
        else:
            raise ValueError('Parameter type should be categorical, ordinal or continuous')

    def f_initial_individual(self):
        ind = dict((nn, self.f_initial_bit_value(p)) for nn, p in self.param_def.iteritems())
        return self.modify_parameters(ind, initial=True)

    @staticmethod
    def f_mate_bit(bit1, bit2, p=None, indpb=0.2):
        if np.random.random() > indpb:
            return bit1, bit2
        if p['type'] == 'continuous':
            # Allow 5% either side
            rng = p['args']
            alpha = 0.05 * (rng[1] - rng[0])
            nbit1, nbit2 = zip(*tools.cxBlend([bit1], [bit2], alpha))[0]
            nbit1 = min(max(nbit1, rng[0]), rng[1])
            nbit2 = min(max(nbit2, rng[0]), rng[1])
            return nbit1, nbit2
        elif p['type'] == 'ordinal':
            a = [aa for aa in p['args'] if min(bit1, bit2) <= aa <= max(bit1, bit2)]
            return np.random.choice(a), np.random.choice(a)
        elif p['type'] == 'categorical':
            return bit2, bit1
        else:
            raise ValueError('Parameter type should be categorical, ordinal or continuous')

    def f_mate(self, x1, x2):
        xm = dict((nn, self.f_mate_bit(x1[nn], x2[nn], self.param_def[nn])) for nn in self.param_def.keys())
        x12 = creator.Individual((k, v[0]) for k, v in xm.iteritems())
        x21 = creator.Individual((k, v[1]) for k, v in xm.iteritems())
        return self.modify_parameters(x12, initial=False), self.modify_parameters(x21, initial=False)

    @staticmethod
    def f_mutate_bit(bit, p=None, mu=0., sigma=0.1, indpb=0.2):
        if p['type'] == 'continuous':
            rng = p['args']
            mu = rng[0] + mu * (rng[1] - rng[0])
            sigma *= rng[1] - rng[0]
            bit = tools.mutGaussian([bit], mu, sigma, indpb)[0][0]
            return min(max(bit, rng[0]), rng[1])
        elif p['type'] == 'ordinal':
            idx = p['args'].index(bit)
            idx += np.random.choice([-1, 0, 1], p=[0.5 * indpb, 1. - indpb, 0.5 * indpb])
            idx = min(max(idx, 0), len(p['args']) - 1)
            return p['args'][idx]
        elif p['type'] == 'categorical':
            if np.random.random() < indpb:
                return np.random.choice(p['args'])
            else:
                return bit
        else:
            raise ValueError('Parameter type should be categorical, ordinal or continuous')

    def f_mutate(self, ind, mu=0., sigma=0.1, indpb=0.2):
        xm = creator.Individual((nn, self.f_mutate_bit(ind[nn], self.param_def[nn], mu=mu, sigma=sigma, indpb=indpb))
                                for nn in self.param_def.keys())
        return self.modify_parameters(xm, initial=False),

    @staticmethod
    def parameter_levels(p, cont_slices=5):
        if p['type'] == 'continuous':
            rng = p['args']
            return list(np.linspace(rng[0], rng[1], cont_slices))
        elif p['type'] == 'ordinal':
            return p['args']
        elif p['type'] == 'categorical':
            return p['args']

    def expand_param_grid(self):
        # Get the discrete parameter levels and return all combinations
        param_lev = [(nn, self.parameter_levels(p, cont_slices=self._cont_slices)) for nn, p in self.param_def.iteritems()]
        param_names, param_lev = zip(*param_lev)

        if np.prod(map(len, param_lev)) > 10000:
            raise ValueError('Configuration will create > 10000 individuals')

        pop = [self.modify_parameters(dict(zip(param_names, x)), True) for x in product(*param_lev)]
        pop = [creator.Individual(ind) for ind in pop]
        return pop

    def evolve(self, mapper=None):
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('min', np.min)
        stats.register('max', np.max)

        toolbox = self.toolbox

        if mapper is not None:
            toolbox.register('map', mapper)

        self.pop, self.logbook = algorithms.eaMuPlusLambda(self.pop, toolbox, self._mu, self._lambda,
                                                           cxpb=self._cxpb, mutpb=self._mutpb,
                                                           ngen=self._ngen, stats=stats, halloffame=self.hof,
                                                           verbose=True)


class StrategyChooseParamsGS(ChooseParamsGA):
    def __init__(self, optimization_name, conn, start_time, end_time, static_params,
                 npop=50, nhof=1, indpb=0.5, tournsize=5, mu=50, lambda_=20, cxpb=0.5,
                 mutpb=0.2, ngen=10, gridsearch=False, backtest_params=None):

        ChooseParamsGA.__init__(self, npop=npop, nhof=nhof, indpb=indpb, tournsize=tournsize,
                                mu=mu, lambda_=lambda_, cxpb=cxpb, mutpb=mutpb, ngen=ngen, gridsearch=gridsearch)
        self._start_time = start_time
        self._end_time = end_time
        self._static_params = static_params
        self._recorder = BackTestMongoOrderRecorder(conn, self.strategy_name(), optimization_name)

        self._score_cache = {}
        self._runs = []

        self._conn = conn

        self._backtest_params = {} if backtest_params is None else backtest_params

    @staticmethod
    def strategy_name():
        """
        The globally unique human readable strategy name
        """
        raise NotImplementedError('Should be implemented by subclass')

    @staticmethod
    def date_ranges():
        """
        The date ranges to cross validate for
        """
        raise NotImplementedError('Should be implemented by subclass')

    @staticmethod
    def out_of_sample():
        """
        The out of sample dates and static parameters
        """
        raise NotImplementedError('Should be implemented by subclass')

    @staticmethod
    def opt_params(short=False):
        """
        The genetic algorithm parameters passed to ChooseParamsGA
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

    def _calculate_score(self, metrics):
        """
        Override to calculate the score using one or more of the standard metrics
        """
        return Scoring.calculate_score_100(metrics)

    @staticmethod
    def get_capital_by_day_as_df(backtest_output):

        risk_dict = {}
        last_day = None
        for time, capital_obj in backtest_output['risk_provider'][backtest_output['risk_provider'].keys()[0]].\
                iteritems():
            current_day = time.date()
            if current_day != last_day:
                capital = capital_obj.capital
                risk_dict[current_day] = capital
                last_day = current_day
        risk_provider_pd = pd.DataFrame(risk_dict.items(), columns=['date', 'capital'])
        return risk_provider_pd

    def get_score(self, backtest_output):
        """
        This is the score used by the GA, defaults to _calculate_score function but override as necessary.
        The simple case is to override _calculate_score to use a different score from the
        metrics or override this method to do something completely different.
        """
        if len(backtest_output['bet_states']):
            bets_df = pd.DataFrame([
               dict_from_bet_state(dt_, bs)
               for dt_, bs in backtest_output['bet_states'].iteritems()
               if bs.status == OrderStatus.SETTLED and bs.matched_odds > 1.
            ])
            risk_provider_pd = StrategyChooseParamsGS.get_capital_by_day_as_df(backtest_output)

            bets_df.loc[:, 'event'] = bets_df.fixture_id
            bets_df.loc[:, 'market'] = bets_df.market_id
            bets_df = pd.concat(apply_betfair_commission([row for _, row in bets_df.iterrows()]), axis=1).transpose()
            bets_df['pnl'] -= bets_df.commission

            metrics = flat_capital_metrics(bets_df, capital=risk_provider_pd)
            if metrics is None:
                return -self.weights[0] * float('inf'), None
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
                'cr_trade': metrics['cr_trade'],
                'hit_ratio': metrics['hit_ratio'],
                'volatility': metrics['volatility (not annualised)'],
                'time_in_market': incr_metrics.time_in_market,
                'rina': incr_metrics.rina,
            }

            metrics['rina'] = incr_metrics.rina

            return self._calculate_score(metrics), metrics_dict
        else:
            return -self.weights[0] * float('inf'), None

    def evaluate(self, ind):
        strategy_params = self._static_params.copy()
        strategy_params.update(ind)

        evaluate_start_time = time.time()

        logging.info(
            'Starting backtest for params %s, start time %s, end time %s' % (
                str(strategy_params), str(self._start_time), str(self._end_time))
        )

        strategy_instance = self.strategy_instance(ind)
        try:
            output = run_backtest(strategy_instance, self._start_time, self._end_time,
                                  **self._backtest_params)
        except:
            logging.error(
                'Exception raised while running backtest for params %s, start time %s, end time %s' % (
                    str(strategy_params), str(self._start_time), str(self._end_time))
            )
            raise

        elapsed = time.time() - evaluate_start_time
        logging.info(
            'Finished backtest for params %s, start time %s, end time %s in %d minutes' % (
                str(strategy_params), str(self._start_time), str(self._end_time), elapsed / 60)
        )

        static_params = self._static_params
        dynamic_params = dict(ind)

        self._recorder.record(
            self._start_time, self._end_time, static_params, dynamic_params, output, datetime.now(pytz.UTC))

        score, metrics = self.get_score(output)

        return score, metrics

    def evaluate_spark(self, ind):
        from sgmtradingcore.analytics.performance import convert_bet_states_to_array
        from sgmtradingcore.backtesting.backtest import run_backtest
        from sgmtradingcore.backtesting.persistence import BacktestOptimizationRun
        from sgmtradingcore.backtesting.recorders import BackTestMetricsCSVRecorder, BackTestMongoOrderRecorder

        return self.evaluate(ind)
