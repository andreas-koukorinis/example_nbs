from itertools import islice, tee, izip

import pandas as pd
import copy

from stratagemdataprocessing.util.dateutils import parse_date_if_necessary
import tqdm

from sgmtradingcore.core.ts_log import get_logger


class PrototypeStrategyRunner(object):

    def __init__(self, strategy_class, strategy_params, run_options=None, feature_runner=None):
        self._run_options = run_options or {}
        self._strategy_class = strategy_class
        self._strategy_params = strategy_params

        self._feature_runner = feature_runner or self.default_feature_runner()

        self._strategy = None
        self._logger = get_logger(self.__class__.__name__)

        self._mp_pool = None
        self._use_mp = False
        if self._run_options.get('n_workers', 1) > 1:
            self._use_mp = True

    @property
    def strategy(self):
        if self._strategy is None:
            self._strategy = self._strategy_class(self._strategy_params, self._feature_runner)
        return self._strategy

    @staticmethod
    def default_feature_runner():
        raise NotImplementedError('Choose default feature runner in the subclass')

    def run(self, start, end):
        """
        Run the strategy from the start time until the end time.
        Will multiprocess if self._run_options['n_workers'] > 1
        :param start: string or datetime
        :param end: string or datetime
        :return: trades from strategy and ticks dataframe
        """
        start_dt = parse_date_if_necessary(start, to_utc=True)
        end_dt = parse_date_if_necessary(end, to_utc=True)

        # Assumes return type of a dataframe
        untransformed_ticks = self.strategy.get_ticks(start_dt, end_dt)
        ticks_df = self.strategy.transform_ticks(untransformed_ticks)

        iter_2, iter_1, iter_0 = tee(ticks_df.itertuples(), 3)

        end_ticks = islice(iter_2, 2, None)
        start_ticks = islice(iter_1, 1, None)

        args = []

        # Queue up the inputs
        for index, (window_end_tick, window_start_tick, current_tick) in enumerate(
                izip(end_ticks, start_ticks, iter_0)):
            window_end = window_end_tick.Index
            window_start = window_start_tick.Index

            params = {'runner_class': self.__class__,
                      'feature_runner': None,
                      'start': window_start,
                      'end': window_end,
                      'params': copy.copy(self.strategy.params),
                      'strategy_class': self.strategy.__class__,
                      'current_tick': current_tick._asdict(),
                      'run_options': self._run_options}

            args.append(params)

        if self._use_mp:
            self._run_multi(args)
        else:
            self._run_single(args)

        return self.strategy.trades, ticks_df

    @staticmethod
    def fetch_and_transform_per_tick_data(feature_runner, start, end, params, strategy_class, current_tick,
                                          run_options=None):
        """
        This is a wrapper around the strategy methods to fetch and transform per-tick data.
        It is done using staticmethods / classmethods / functions to allow us to parallelize the computation

        See those function for meanings of input variables

        If the current_tick has a field 'fetch_per_tick_data' == False then we skip this.

        The local imports are to ensure we get the global feature runner correctly in a multiprocessing environment.

        :param run_options: options affecting the execution (not the result) of collecting data
        :param current_tick: a dict (row of a dataframe)
        :param strategy_class: the class of the strategy
        :param params: the parameters for the strategy
        :param end: the end datetime for the current interval
        :param start: the start datetime for the current interval
        :param feature_runner : feature runner
        :return Transformed per_tick_data (can be any picklable object)
        """

        from sgmtradingcore.analytics.features.runner import parallel_runner

        if not current_tick.get('fetch_per_tick_data', True):
            return

        feature_runner = feature_runner or parallel_runner
        features = strategy_class.fetch_per_tick_data(start, end, params, feature_runner)
        transformed_per_tick_data = strategy_class.transform_per_tick_data(current_tick,
                                                                           features,
                                                                           params)

        return transformed_per_tick_data

    def _run_single(self, args):

        for arg in tqdm.tqdm(args):
            arg['feature_runner'] = self._feature_runner
            transformed_per_tick_data = fetch_and_transform_per_tick_data(arg)

            window_start = arg['start']
            window_end = arg['end']
            current_tick = arg['current_tick']
            self.strategy.generate_trades(current_tick,
                                          transformed_per_tick_data,
                                          window_start,
                                          window_end)

        return self.strategy.trades

    def _initialize_pool(self, run_options):
        return self._feature_runner.initialize_mp_pool(run_options['n_workers'], maxtasksperchild=100)

    def _run_multi(self, args):

        pool = self._initialize_pool(self._run_options)

        async_results = [pool.apply_async(fetch_and_transform_per_tick_data, (p,)) for p in args]

        for params, async_result in tqdm.tqdm(zip(args, async_results)):
            # Ensure we process in the right order
            transformed_per_tick_data = async_result.get()

            window_start = params['start']
            window_end = params['end']
            current_tick = params['current_tick']
            self.strategy.generate_trades(current_tick,
                                          transformed_per_tick_data,
                                          window_start,
                                          window_end)

        pool.close()


def fetch_and_transform_per_tick_data(args):
    runner_class = args['runner_class']
    feature_runner = args['feature_runner']
    return runner_class.fetch_and_transform_per_tick_data(feature_runner, args['start'], args['end'], args['params'],
                                                          args['strategy_class'], args['current_tick'],
                                                          run_options=args['run_options'])


class PrototypeStrategy(object):
    """
    Strategy to perform prototype backtests on market features / tsdb.

    Design:

    We have a low-frequency signal which is given by the datafromes produced by get_ticks -> transform_ticks
    (i.e. candles)

    For the interval between two ticks (e.g. candles) (which corresponds to a row of the above dataframe) we wish to
    process high frequency data (e.g. trades / lob) which are given by fetch_per_tick_data -> transform_per_tick_data.

    Transform_per_tick_data should reduce the high frequency data to the minimum needed to make trading decisions and must
    be done in a fashion that can be run in parallel.
    Calls to fetch_per_tick_data / transform_per_tick_data can only be a function of the current tick and the strategy
    input parameters i.e. they can't be a function of the state of the strategy - any logic that requires this must go
    in the generate_trades function.

    The generate_trades function runs in sequence, taking the current tick, and one set of the transformed features
    and saves any trades in self._trades. As this cannot be parallelised easily, it has access to instance variables
    so any state needed can be saved on the strategy object itself
    """

    def __init__(self, params, feature_runner):
        self._feature_runner = feature_runner
        self._params = params
        self._trades = []

    def reset(self):
        self._trades = []

    @property
    def params(self):
        return self._params

    def transform_ticks(self, ticks):
        """
        The strategy ticks once per row of this dataframe.

        Takes as input the output of the features specified by get_ticks.

        Has to be indexed by time.
        :param ticks:
        :return:
        """
        return ticks

    def get_ticks(self, start, end):
        """
        Retrieves the raw data needed to calculate ticks
        :param start:
        :param end:
        :return:
        """
        raise NotImplementedError('Implement in subclass')

    @classmethod
    def feature_requests(cls, params):
        """
        This is a list of tickers and feature requests we need to fetch for in between each tick interval.

        :return: [feature_requests]
        """
        raise NotImplementedError('List of features')

    @classmethod
    def fetch_per_tick_data(cls, start_dt, end_dt, params, runner):
        """
        Specifies how to fetch the the data needed to make trading decisions over a time-window
        Output is passed to transform_per_tick_data

        Should return open-close interval

        Is a class method for parallelisation, so has no access to instance variables
        :param start_dt:
        :param end_dt:
        :param params:
        :param runner:
        :return:
        """
        raise NotImplementedError('Implement in subclass')

    @classmethod
    def transform_per_tick_data(cls, current_tick, per_tick_data, params):
        """
        We transform the data returned by fetch_per_tick_data according to this method.
        You should reduce the features to the minimum needed for trading decisions in this method to avoid overhead
        when sending back to the main process to generate trades.

        Is a class method for parallelisation, so has no access to instance variables
        :param current_tick:
        :param per_tick_data:
        :param params:
        :return:
        """
        return per_tick_data

    def generate_trades(self, current_tick, transformed_per_tick_data, start, end):
        """
        Run in the main thread as is history dependent. Can access
        :param current_tick: one row from the output of transform_ticks
        :param transformed_per_tick_data: the output of transform_per_tick_data for the current tick
        :param start: start of the current tick interval (i.e. the end of the period that the current tick covers)
        :param end: end of the current tick interval (i.e. the time at which the next tick will occur)
        :return: None - trades should be appended to self._trades
        """
        raise NotImplementedError('Implement in subclass')

    @property
    def trades(self):
        return self._trades


class PrototypeTrade(object):
    def __init__(self, entry_time, entry_price, is_long, exit_time=None, exit_price=None):
        self._entry_time = entry_time
        self._entry_price = entry_price
        self._is_long = is_long
        self._exit_time = exit_time
        self._exit_price = exit_price
        self._pnl = None
        self._is_open = True

    @property
    def entry_time(self):
        return self._entry_time

    @property
    def entry_price(self):
        return self._entry_price

    @property
    def exit_time(self):
        return self._exit_time

    @property
    def exit_price(self):
        return self._exit_price

    @property
    def is_long(self):
        return self._is_long

    def exit_trade(self, price, time):
        self._exit_price = price
        self._exit_time = time
        self._is_open = False
        self._pnl = None
        # and calculate pnl

    @property
    def is_open(self):
        return self._is_open

    @property
    def pnl(self):
        if self._pnl is not None:
            return self._pnl

        if self._exit_price is not None:
            if self._is_long:
                self._pnl = self._exit_price - self._entry_price
            else:
                self._pnl = self._entry_price - self._exit_price

        else:
            self._pnl = 0.0
        return self._pnl

    def to_dict(self):
        fields = ['entry_price',
                  'entry_time',
                  'exit_price',
                  'exit_time',
                  'is_long',
                  'pnl',
                  'is_open']
        return {f: getattr(self, f) for f in fields}

    @staticmethod
    def from_dict(d):
        trade = PrototypeTrade(d['entry_time'], d['entry_price'], d['is_long'], d['exit_time'], d['exit_price'])
        if trade.pnl != d['pnl']:
            raise ValueError('Input PnL is {}, Output PnL is {}'.format(trade.pnl, d['pnl']))
        elif trade.is_open != d['is_open']:
            result_str = 'not' if not trade.is_open else ''
            input_str = 'not' if not d['is_open'] else ''
            raise ValueError('Input is {} open, resulting trade is {} open'.format(input_str, result_str))

        return trade

    def __str__(self):
        exit_price = '{:2.2f}'.format(self.exit_price) if self.exit_price is not None else 'None'
        return 'CryptoPrototypeTrade({s.entry_price:2.2f}, {exit_price}, {s.entry_time}, {s.exit_time}, {s.is_long})'.format(
            s=self, exit_price=exit_price)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def to_dataframe(trades):
        if not isinstance(trades, list):
            trades = [trades]

        return pd.DataFrame([t.to_dict() for t in trades])
