import logging
import os
from collections import defaultdict
from datetime import datetime
from functools import partial
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML
from dateutil.relativedelta import relativedelta
from matplotlib import rcParams
from statsmodels.graphics.tsaplots import plot_acf

from sgmtradingcore.analytics.metrics import aggr_trades, get_liability
from stratagemdataprocessing.util.cache_data import RollingCacheDataFramePickle
from sgmtradingcore.analytics.metrics import financial_metrics
from sgmtradingcore.analytics.returns.util import get_traces, get_aggr_returns, \
    display_df_style, f_order_to_row, f_instruction_to_row, \
    custom_notebook_display, StrategySettings
from sgmtradingcore.backtesting.persistence import MongoStrategyHelper
from sgmtradingcore.core.trading_types import InstructionStatus, OrderStatus

pd.set_option('display.max_rows', 500)


class NoOrdersFoundException(Exception):
    pass


class NoInstructionsFoundException(Exception):
    pass


class StrategiesBacktestAnalysis(object):
    """
    Help analysing returns from backtest. 
    Add the results you want using add_backtests_results,
        and show stats grouped by different things using the available methods
    
    Most interesting features include:
        - loading backtest results from different backtest runs
        - adding fixtures information to all your instructions easily
        - wrapper around the usual functions financial_metrics and
            flat_capital_metrics 
        - cache which saves in files the first time it loads the backtest results

    Improvement include specifying it's own f_instruction_to_row function 
        to extract information strategy specify
    
    More work to plug in realtime capital allocation and changing capital
        in the branch trading_metrics in 
        sgmtradingcore.analytics.capital_allocation.strategy
        
    Attributes
    ----------
    start_date: cf init docstring
    end_date: cf init docstring
    common_comps: cf init docstring
    use_cache: cf init docstring

    all_instructions: (dict) (tuple) backtest_key -> (DataFrame) instructions

    _concat_instr_df: attribute for property concat_instr_df
        (DataFrame) concatenation of instructions in all_instructions
    _fixtures_info: attribute for property fixtures_info
        (DataFrame) fixtures information from the fixtures_info_fun property
    display_fun: (fun) function which takes an HTML and display
        - at the moment it either uses display from IPython.display 
            or does not display anything at all
    _logger: attribute for property logger
    _log_lvl: cf init docstring
    sh: (MongoStrategyHelper) instance to get backtest results
    
    Examples
    -----------
    First example with implemented subclass in sgmfootball.backtest.return_analysis
    
    Second example once the abstract methods are implemented:
        start_date, end_date = '2016-09-08', '2017-02-01'
        sa = StrategiesBacktestAnalysis(start_date, end_date, verbose=True)
        ss = StrategyBacktestSettings('Analyst_FTAHG', 'all', None, 'capital_allocation',
                                         PROD_ALGO_SPORTS_CONFIG['trading_user_id'])
        sa.add_backtests_results(ss)
        sa.display_metrics(['market', 'comp'])
    
    """

    def __init__(self, start_date, end_date, common_comps,
                 verbose=True, log_lvl=logging.INFO, use_cache=True):
        """
        Parameters
        ----------
        start_date: (str) start date of the instructions to take from backtest
        end_date: (str) end date of the instructions to take from backtest
        common_comps: competitions to get from the backtests
        verbose: whether to display the statistics or just return them
        log_lvl: logging level (should be like logging.DEBUG or logging.WARNING)
        """
        if common_comps is None:
            raise ValueError('Please specify the list of competitions to get from'
                             'fixtures_info_fun')
        self.start_date = start_date
        self.end_date = end_date
        self.common_comps = common_comps

        self.use_cache = use_cache  # whether to use cache for instructions for example

        self.all_instructions = {}

        self._concat_instr_df = None
        self._fixtures_info = None

        # Logs and comments
        self.display_fun = custom_notebook_display if verbose else lambda x: ''
        self._logger = None
        self._log_lvl = log_lvl

        self.sh = MongoStrategyHelper()
        self.sh._logger.set_level(self._log_lvl)

    @property
    def sport(self):
        """
        Returns
        -------
        sport: (Sports) current sport.
          - Sports is the enum from stratagemdataprocessing/enums/odds.py
        """
        raise NotImplementedError('Should be implemented in subclass')

    def fixtures_info_fun(self):
        """
        Returns
        -------
        Function with argument parameters start_date, end_date and common_comps.
         which returns a DataFrame with fixtures info between start_date and
         end_date for competition comp. it should have the columns self.fixtures_info_cols
        """
        raise NotImplementedError('Should be implemented in subclass')

    @property
    def fixtures_info_cols(self):
        """
        Returns
        -------
        (list) columns returned by fixtures_info_fun.
        WARNING : fixture_id should be in the list as it is 
        used as an id to match up fixtures info with the instructions
        
        """
        raise NotImplementedError('Should be implemented in subclass')

    def add_backtests_results(self, ss, capital,
                              use_orders_for_instrs=False,
                              kick_off_as_dt=False,
                              extra_cols=None, extra_info=None):
        """
        Add the backtest results of the strategy setting, given a capital

        Parameters
        ----------
        ss: (StrategySettings) of the backtest to add
        capital: (int) or (None) the capital that was used in the backtest
             THIS CAPITAL PART IS IMPORTANT TO HAVE A MEANINGFUL IDEA OF
             WHAT THE EXPECTED RETURN SHOULD BE !
        use_orders_for_instrs: (bool) whether to use orders instead of instructions
            This will just assume that the orders_df are instructions. It is fine but
            the names of the functions and variables could be misleading
        kick_off_as_dt: (boolean) whether to change the placed time to be the kick off time
            This can be useful when having long positions and we want instructions
            from the same fixture to be accounted for in the same daily return
        extra_cols: (dict) what extra columns to add to the DataFrame of instructions
        extra_info: (dict) extra information to take from raw instructions, e.g.
            {'timestamp': 'details.signals.0.timestamp'}
        
        Returns
        -------
        Nothing, it adds the results to the dict attribute self.all_instructions
        """
        if ss.key in self.all_instructions:
            raise Exception('{} has already been added'.format(ss.key))

        self._reset_concat_data()
        if self.use_cache:
            fun_params = {'ss': ss, 'capital': capital,
                          'use_orders_for_instrs': use_orders_for_instrs,
                          'kick_off_as_dt': kick_off_as_dt,
                          'extra_info': extra_info}
            cache_settings = self.cache_settings(fun_params=fun_params)
            instr_cache = RollingCacheDataFramePickle(**cache_settings)
            instr_cache.logger.setLevel(self._log_lvl)
            instr_df = instr_cache.get_data(datetime.strptime(self.start_date, '%Y-%m-%d'),
                                            datetime.strptime(self.end_date, '%Y-%m-%d'))
        else:
            instr_df = self._get_formatted_instructions(
                ss, capital, use_orders_for_instrs, kick_off_as_dt,
                extra_info=extra_info)
        if not instr_df.empty:
            idx_pnl_nan = instr_df['pnl'].isnull()
            if sum(idx_pnl_nan) > 0:
                self.logger.warning('Getting rid of {}/{} instructions without pnl from orders'.format(
                    sum(idx_pnl_nan), len(instr_df)))
                instr_df = instr_df.loc[~idx_pnl_nan]
            #
            if extra_cols is not None:
                for k, v in extra_cols.iteritems():
                    instr_df[k] = v
            self.all_instructions[ss.key] = instr_df.reset_index(drop=True)

    def display_metrics(self, grp_by, remove_size_null=True, remove_adj_cap_nan=False,
                        ret_aggr_by='date', trade_agg_periods=None,
                        trading_periods=None, target_vol=None,
                        pre_group_by=None, how_pre_group_capital=None,
                        return_df=False):
        """
        Display the flat_capital_metrics stats for the loaded instructions

        Parameters
        ----------
        grp_by: (list of str) grouping columns in the financial_metrics function
        remove_size_null: (bool) whether to remove the instructions with no stake
        remove_adj_cap_nan: (bool) whether to remove the instructions with no adjustment capital 
            you need to have used the function adjust_capital for this to make sense or to have the column
            adj_capital to be able to use this argument
        Returns
        -------
        nothing, it displays the table metrics
        """
        instr_df = self._get_instr_after_remove(remove_size_null,
                                                remove_adj_cap_nan)
        extra_unique_cols = list(grp_by) if grp_by is not None else []
        if ret_aggr_by != 'date':
            extra_unique_cols.append(ret_aggr_by)

        instr_df, metrics_odds_col = self._pre_group_instr(instr_df,
                                                           pre_group_by,
                                                           how_pre_group_capital,
                                                           extra_unique_cols)

        fmc = np.round(financial_metrics(instr_df, ret_aggr_by,
                                         groupby=grp_by,
                                         trade_agg_periods=trade_agg_periods,
                                         trading_periods=trading_periods,
                                         target_vol=target_vol,
                                         odds_col=metrics_odds_col,
                                         dt_col='dt'), 3)
        show_metrics = [c for c in self.show_metrics if c in fmc.columns]
        met_df = fmc[show_metrics].rename(columns=self.rename_metrics)
        self.display_fun(met_df)

        if return_df:
            return met_df

    def _pre_group_instr(self, instr_df, pre_group_by, how_pre_group_capital,
                         extra_unique_cols):
        metrics_odds_col = 'odds'
        if pre_group_by is not None:
            unique_is_back = instr_df.groupby(pre_group_by).apply(
                lambda x: len(x['is_back'].unique()) == 1).all()
            if unique_is_back:  # only all back or all lay for same pre-groupby
                instr_df = self._pre_group_back_instr(instr_df, pre_group_by,
                                                      how_pre_group_capital,
                                                      extra_unique_cols=extra_unique_cols)
            else:  # average odds would not be correct for example
                self.logger.warning('Column is_back is not unique by {} so the'
                                    ' odds will not be available after pre-'
                                    'grouping instrs'.format(pre_group_by))
                metrics_odds_col = None  # do not need the odds as we add liability
                instr_df = instr_df.copy()
                instr_df['liability'] = instr_df.apply(get_liability, axis=1)
                del instr_df['odds']
                instr_df = self._pre_group_back_lay_instr(instr_df, pre_group_by,
                                                          how_pre_group_capital,
                                                          extra_unique_cols=extra_unique_cols)
        return instr_df, metrics_odds_col

    def _pre_group_back_instr(self, instr_df, pre_group_by, how_pre_group_capital,
                              extra_unique_cols=None):
        aggr_cols = pre_group_by
        mean_cols = []
        stake_mean_cols = ['odds']
        first_cols = ['dt']
        sum_cols = ['pnl', 'stake']
        unique_cols = ['is_back']
        if extra_unique_cols is not None:
            unique_cols += extra_unique_cols
        stake_col = 'stake'
        self._add_capital_to_cols(how_pre_group_capital, mean_cols,
                                  sum_cols, unique_cols)

        instr_df = aggr_trades(instr_df, aggr_cols, mean_cols=mean_cols,
                               sum_cols=sum_cols,
                               first_cols=first_cols,
                               unique_cols=unique_cols,
                               stake_weighted_mean_cols=stake_mean_cols,
                               stake_col=stake_col)
        return instr_df

    def _pre_group_back_lay_instr(self, instr_df, pre_group_by, how_pre_group_capital,
                              extra_unique_cols=None):
        aggr_cols = pre_group_by
        mean_cols = []
        stake_mean_cols = []  # no odds columns here for back lay
        first_cols = ['dt', 'is_back']
        sum_cols = ['pnl', 'stake', 'liability']
        unique_cols = []
        if extra_unique_cols is not None:
            unique_cols += extra_unique_cols
        stake_col = 'stake'
        self._add_capital_to_cols(how_pre_group_capital, mean_cols,
                                  sum_cols, unique_cols)

        instr_df = aggr_trades(instr_df, aggr_cols, mean_cols=mean_cols,
                               sum_cols=sum_cols, first_cols=first_cols,
                               unique_cols=unique_cols,
                               stake_weighted_mean_cols=stake_mean_cols,
                               stake_col=stake_col)
        return instr_df

    def _add_capital_to_cols(self, how_pre_group_capital, mean_cols, sum_cols,
                             unique_cols):
        if how_pre_group_capital is None or how_pre_group_capital not in [
            'mean', 'sum', 'unique']:
            raise ValueError('If pre-grouping trades, you should specify'
                             'how to group capital (either mean or sum)')
        else:
            if how_pre_group_capital == 'mean':
                mean_cols.append('capital')
            elif how_pre_group_capital == 'sum':
                sum_cols.append('capital')
            elif how_pre_group_capital == 'unique':
                unique_cols.append('capital')
            else:
                raise Exception('how_pre_group_capital value incorrect')

    def display_trace(self, grp_by, ret_aggr_by='date', remove_size_null=True,
                      remove_adj_cap_nan=False, row_hue_col=None):
        """
        Display the cumulative return trace for the loaded instructions

        Parameters
        ----------
        grp_by: 
        ret_aggr_by: (str) how to aggregate the returns
        remove_size_null: (bool) whether to remove the instructions with no stake
        remove_adj_cap_nan: (bool) whether to remove the instructions with no adjustment capital 
            you need to have used the function adjust_capital for this to make sense or to have the column
        adj_capital to be able to use this argument
        row_hue_col: list 3 elements to tell what columns to put as row, hue and col in the seaborn facetGrid
            e.g. if grp_by=['a', 'b'], and you want 'a' in hue and 'b' in col and nothing in row : [None, 0, 1]

        Returns Nothing, it plots the trace
        -------

        """
        instr_df = self._get_instr_after_remove(remove_size_null, remove_adj_cap_nan)
        self.static_display_trace(instr_df, grp_by, ret_aggr_by, row_hue_col)


    def display_return_on_stake_histogram(self, ret_aggr_by,
                                          remove_size_null=True,
                                          remove_adj_cap_nan=False):

        instr_df = self._get_instr_after_remove(remove_size_null=remove_size_null,
                                                remove_adj_cap_nan=remove_adj_cap_nan)
        self.return_on_stake_hist(instr_df, ret_aggr_by)

    @staticmethod
    def return_on_stake_hist(instr_df, ret_aggr_by):
        returns = instr_df.groupby(ret_aggr_by)['pnl'].sum() / \
                  instr_df.groupby(ret_aggr_by)['stake'].sum()
        rcParams['figure.figsize'] = (16, 6)
        sns.distplot(returns, rug=True, hist=True, rug_kws={"color": "g"},
                     kde_kws={"color": "k", "lw": 1})
        plt.axvline(0., color='grey', linestyle='--')  # add line
        q_pos_ = plt.gca().get_ylim()[1] * 0.95
        for q in [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]:
            q_ = returns.quantile(q)
            plt.axvline(q_, color='red', linestyle='--')  # add line
            plt.text(q_, q_pos_, '{}pct'.format(int(q * 100)), color='red')
        plt.xlabel('return on stake (pnl/stake)')
        plt.ylabel(ret_aggr_by + ' density')
        plt.title('histogram of ' + ret_aggr_by + ' returns')
        plt.show()

    @staticmethod
    def static_display_trace(instr_df, grp_by, ret_aggr_by, row_hue_col):
        traces = get_traces(instr_df, grp_by, ret_aggr_by)
        if row_hue_col is None:
            if grp_by is not None:
                row = grp_by[-1]
                hue = None if len(grp_by) <= 1 else grp_by[-2]
                col = None if len(grp_by) <= 2 else grp_by[-3]
            else:
                hue, row, col = None, None, None
        else:
            row = grp_by[row_hue_col[0]] if row_hue_col[
                                                0] is not None else None
            hue = grp_by[row_hue_col[1]] if row_hue_col[
                                                1] is not None else None
            col = grp_by[row_hue_col[2]] if row_hue_col[
                                                2] is not None else None
        if col is not None:
            size, aspect = 2.75, 1.75
        else:
            size, aspect = 5.5, 3.5
        g = sns.FacetGrid(traces.reset_index(),
                          hue=hue,
                          row=row,
                          col=col,
                          size=size, aspect=aspect,
                          sharex=True,
                          sharey=False,
                          )
        g.map(plt.plot, ret_aggr_by, 'cum_return')
        g.add_legend()
        plt.show()

    def get_returns(self, grp_by, ret_grp_by):
        aggr_returns = self.concat_instr_df.groupby(grp_by).apply(
            get_aggr_returns, ret_grp_by=ret_grp_by)
        if not isinstance(aggr_returns, pd.DataFrame):
            aggr_returns = aggr_returns.unstack()
        return aggr_returns

    @property
    def concat_instr_df(self):
        """
        DataFrame containing all the instructions from all the added backtest
        results
        """
        if self._concat_instr_df is None and len(self.all_instructions) > 0:
            self._concat_instr_df = pd.concat(self.all_instructions.values(), axis=0)
        return self._concat_instr_df

    @property
    def fixtures_info(self):
        if self._fixtures_info is None:
            self._fixtures_info = self.fixtures_info_fun()(
                self.start_date, self.end_date, comp=self.common_comps)
        return self._fixtures_info

    @property
    def logger(self):
        if self._logger is None:
            logging.basicConfig()
            self._logger = getLogger(__name__)
            self._logger.setLevel(self._log_lvl)
        return self._logger

    def display_corr(self, grp_by, corr_distinct=0, ret_grp_by='date'):
        """
        Display the correlation matrix

        Parameters
        ----------
        grp_by: (list of str) how to group the results
        corr_distinct: (int) from what column index
            the correlation matrix should be distinct in the display.
        ret_grp_by: how to aggregate the returns of the instructions

        Returns
        -------
        the aggregated returns used to compute the correlation matrix coefficients 
        """
        self.display_fun(HTML('<h4> Correlation analysis on {} returns</h4>'.format(ret_grp_by)))

        aggr_returns = self.concat_instr_df.groupby(grp_by).apply(
            get_aggr_returns, ret_grp_by=ret_grp_by)
        for corr_name in ['pearson', 'kendall', 'spearman']:
            self.display_fun(HTML('<h5>Correlation table using {} method</h4>'.
                                  format(corr_name)))
            if not isinstance(aggr_returns, pd.DataFrame):
                aggr_returns = aggr_returns.unstack()
            table_corr = np.round(aggr_returns.T.corr(method=corr_name), 2)
            if corr_distinct > 0:
                index_vals = table_corr.index.values
                if corr_distinct == 1:
                    distinct_index = list(set([c[corr_distinct - 1]
                                               for c in index_vals]))
                else:
                    distinct_index = list(set([c[:corr_distinct]
                                               for c in index_vals]))
                for di in distinct_index:
                    display_df_style(table_corr.loc[di, di].dropna(how='all'),
                                     'Looking at {}'.format(di), -1, 1)
            else:
                display_df_style(table_corr.dropna(how='all'), '', -1, 1)
        return aggr_returns

    def display_autocorr(self, grp_by, ret_aggr_by, lags=7):
        """
        Display the basic auto-correlation coefficient for a timeseries of returns.
        It is simply a wrapper around plot_acf from statsmodels.graphics.tsaplots

        Parameters
        ----------
        grp_by:
        ret_aggr_by
        lags: max lag of the autocorrelation on the plot.

        Returns
        -------

        """
        self.display_fun(HTML('<h4> Auto-correlation analysis on {} returns</h4>'.
                              format(ret_aggr_by)))

        def _display_autocorr(df_, ret_aggr_by_):
            aggr_returns = get_aggr_returns(df_, ret_aggr_by_)
            rcParams['figure.figsize'] = 17, 5
            plot_acf(aggr_returns, lags=lags)
            plt.show()

        df = self.concat_instr_df.set_index(grp_by)
        for index_val in set(df.index.values):
            self.display_fun(HTML('<h6> Auto-correlation for {}</h6>'.
                                  format(index_val)))
            _display_autocorr(df.loc[index_val], ret_aggr_by)

    def _reset_concat_data(self):
        self._concat_instr_df = None

    @property
    def show_metrics(self):
        """
        Returns the list of the columns of the backtest metrics to keep
        """
        return ['n_trades', 'sharpe_ratio', 'hit_ratio', 'cr_trade', 'cr_day',
                'maximum_drawdown', 'maximum_runup', 'cum_return',
                'unitary_stake_return', 'volatility (not annualised)',
                'total_pnl', 'scale', 'volatility_annualised']

    @property
    def rename_metrics(self):
        """
        Returns the rename dictionary in you need to rename columns of the backtest metrics
        """
        return {'maximum_drawdown': 'max_DD', 'maximum_runup': 'max_runup',
                'unitary_stake_return': 'unit_stake_return',
                'volatility (not annualised)': 'vol_not_ann',
                'volatility_annualised': 'vol_annualised',
                'total_pnl': 'pnl'}

    def _get_formatted_instructions(self, ss, capital, use_orders_for_instrs,
                                    kick_off_as_dt, extra_info=None):
        sh = self.sh
        fixtures_info_cols = self.fixtures_info_cols
        start_date, end_date = self.start_date, self.end_date
        logger = self.logger
        fixtures_info = self.fixtures_info
        sport = self.sport
        use_orders_for_instrs = use_orders_for_instrs
        instr_df = self.static_get_formatted_instructions(
            fixtures_info, logger, sh, fixtures_info_cols, sport,
            start_date, end_date, ss, capital,
            use_orders_for_instrs, kick_off_as_dt, extra_info=extra_info)
        return instr_df

    def adjust_instructions_capital(self, capital_ts, total_capital_ts=None):
        """
        adjust the instructions dataframe to take into account the capital allocation in capital_ts
        Parameters
        ----------
        capital_ts: (dict) strategy_descr -> (dict) dt -> capital specific (float)
        assumption that the capital is a split of the initial capital specified in the initial DataFrame

        Returns 
        -------
        Nothing, it updates the stake, pnl, size_wanted and of the attribute _concat_instr_df
        it also adds the columns : 
            - capital_adjusted: in case we want to adjust with another capital_ts
            - adj_cap: filled with 1. and NaN to know the fixtures which had capital allocated in capital_ts   

        WARNING assumes that strategy_desc is unique across strategies too ! 
        for example FFM_FTAHG could not have same strategy desc as Analyst_FTAHG
        """
        if not isinstance(capital_ts, dict):
            raise Exception('adjust_capital should be a dictionary of comp -> capital_ts')

        no_instrs = len(self.concat_instr_df)

        if 'capital_adjusted' not in self._concat_instr_df:
            self._concat_instr_df['capital_adjusted'] = self._concat_instr_df['capital'].copy()

        # calculate the total capital allocated at each point in time
        if total_capital_ts is None:
            total_capital_ts = defaultdict(lambda: 0)
            for sd, cap_ts in capital_ts.iteritems():
                for d, c in cap_ts.iteritems():
                    total_capital_ts[d] += c
        self._concat_instr_df['capital'] = self.concat_instr_df.apply(
            lambda x: total_capital_ts[x['dt'].date()], axis=1)  # that's the capital we allocated at each cluster

        self._concat_instr_df.loc[:, 'adj_cap'] = self._concat_instr_df.apply(
            lambda x: capital_ts.get(x['strategy_desc'], {}).get(x['dt'].date(), np.nan) /
                      x['capital_adjusted'], axis=1)

        idx_zero_cap = self._concat_instr_df.loc[:, 'adj_cap'] == 0
        if sum(idx_zero_cap) > 0:
            # this is so that we can recall adjust_instructions capital with another timeseries
            # without having to reload the instructions. The stake pnl etc.. will be small and not 0
            cap_min = 100
            self.logger.warning('Setting capital 0 to {} for {}/{} instructions for coding reasons'.format(
                cap_min, sum(idx_zero_cap), len(self._concat_instr_df)))
            self._concat_instr_df.loc[idx_zero_cap, 'adj_cap'] = self._concat_instr_df.loc[idx_zero_cap].apply(
                lambda x: cap_min / x['capital_adjusted'], axis=1)

        idx_nan = self._concat_instr_df['adj_cap'].isnull()
        if sum(idx_nan) > 0:
            self.logger.warning('{}/{} instructions with unknown adjustment capital will not be adjusted'
                                .format(sum(idx_nan), no_instrs))

        self._concat_instr_df.loc[~idx_nan, 'pnl'] *= self._concat_instr_df.loc[~idx_nan, 'adj_cap']
        self._concat_instr_df.loc[~idx_nan, 'stake'] *= self._concat_instr_df.loc[~idx_nan, 'adj_cap']
        self._concat_instr_df.loc[~idx_nan, 'size_wanted'] *= self._concat_instr_df.loc[~idx_nan, 'adj_cap']
        self._concat_instr_df.loc[~idx_nan, 'capital_adjusted'] *= self._concat_instr_df.loc[~idx_nan, 'adj_cap']
        # set adj_cap to 1 because the capital has been adjusted
        self._concat_instr_df.loc[~idx_nan, 'adj_cap'] /= self._concat_instr_df.loc[~idx_nan, 'adj_cap']


    @staticmethod
    def static_get_formatted_instructions(
            fixtures_info, logger, sh, fixtures_info_cols, sport,
            start_date, end_date, ss, capital,
            use_orders_for_instrs, kick_off_as_dt, extra_info=None):
        """
        Notes
        -----
        Instructions which did not generate any orders are excluded
        Instructions PnL is taken from orders (if use_orders_for_instrs is False)
        
        """
        try:
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')

            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')

            instructions, orders = sh.get_backtest_result_multiple_days(
                ss.strategy_name, ss.strategy_desc, ss.trading_user_id, ss.strategy_code,
                start_date, end_date, mnemonic=ss.mnemonic)
            # Keeping ClOSED instructions and SETTLED orders

            instructions = [i for i in instructions
                            if i['status'] == InstructionStatus.CLOSED
                            and 'id' in i]

            logger.info('{} | Keeping only SETTLED orders'.format(ss))
            orders = [o for o in orders if o['status'] == OrderStatus.SETTLED]
            # Convert list of orders / instructions into DataFrames
            order_to_row_fun = partial(f_order_to_row, sport)
            orders_df = pd.DataFrame(map(order_to_row_fun, orders))
            if orders_df.empty:
                raise NoOrdersFoundException('{} : {}-{}'.format(ss, start_date, end_date))
            if use_orders_for_instrs:
                instr_df = orders_df
            else:
                intr_to_row_fun = partial(f_instruction_to_row, sport,
                                          extra_info=extra_info)
                instr_df = pd.DataFrame(map(intr_to_row_fun, instructions))
                if instr_df.empty:
                    raise NoInstructionsFoundException(
                        '{} : {}-{}'.format(ss, start_date, end_date))

                # Add PnL from orders
                pnl_instr = orders_df.groupby('instruction_id').sum()['pnl'].reset_index()
                instr_df = instr_df.merge(pnl_instr, how='left', on='instruction_id')

            # Add information such as competition and kick off
            no_instr = len(instr_df)
            instr_df = instr_df.merge(fixtures_info[fixtures_info_cols], on='fixture_id')
            if len(instr_df) < no_instr:
                logger.warning("Removing {}/{} instrs that were not in fixtures_info ! "
                               "Check that common_comps contains the competition you want !".
                               format(no_instr - len(instr_df), no_instr))
            if instr_df.empty:
                raise NoInstructionsFoundException('{} - {}-{}'.format(ss, start_date, end_date))

            if use_orders_for_instrs:  # for orders
                rename_cols = {
                    'matched_odds': 'odds',
                    'matched_size': 'stake',
                }
            else:  # for instructions
                rename_cols = {
                    'average_price_matched': 'odds',
                    'size_matched': 'stake',
                }
            instr_df = instr_df.rename(columns=rename_cols)
            if kick_off_as_dt:
                logger.info('Setting date of the instructions as kick off time '
                            'to group metrics the same way across strategies')
                if 'dt' in instr_df.columns and 'kick_off' in instr_df.columns:
                    del instr_df['dt']
                instr_df = instr_df.rename(columns={'kick_off': 'dt'})
            else:
                if 'dt' not in instr_df.columns:
                    raise ValueError('No dt extracted from the instruction')

            instr_df = instr_df.loc[instr_df['odds'] > 1.]
            instr_df.loc[:, 'capital'] = capital

            # add different formatting from dt
            instr_df.loc[:, 'month'] = instr_df['dt'].apply(lambda x: x.strftime('%m'))
            instr_df.loc[:, 'year'] = instr_df['dt'].apply(lambda x: x.strftime('%Y'))
            instr_df.loc[:, 'y_month'] = instr_df.apply(lambda x: x['year'] + '_' + x['month'], axis=1)
            instr_df.loc[:, 'y_week'] = instr_df['dt'].apply(lambda x: x.strftime('%Y-%V'))

            # Add information from the stratagem settings
            for ss_k, ss_v in ss.__dict__.iteritems():
                instr_df.loc[:, ss_k] = ss_v

        except NoInstructionsFoundException:
            instr_df = pd.DataFrame()
        except NoOrdersFoundException:
            instr_df = pd.DataFrame()
        return instr_df

    @property
    def cache_dir(self):
        return os.path.expanduser('~/.rolling_cache')

    def cache_settings(self, fun_params=None):
        return {'cache_name': 'capital_allocation',
                'rolling_delta': relativedelta(days=7),
                'utd_delta': relativedelta(days=7),
                # the parameters harcd set in the partial function
                # are not in the cache_params as it is assumed
                # that it will not change or does change the data
                'fun': partial(self.static_get_formatted_instructions,
                               self.fixtures_info, self.logger, self.sh,
                               self.fixtures_info_cols, self.sport),
                'fun_params': fun_params,
                'working_dir': self.cache_dir,
                'delta_cacheable': relativedelta(days=2),
                'code_version': None,
                'check_rolling_friendly': 3,
                'date_col': 'dt',
                'drop_duplicates_subset': ['instruction_id']
                }

    def _get_instr_after_remove(self, remove_size_null, remove_adj_cap_nan):
        """
        Filters some of the instructions based on the stake and adj_cap
        
        Parameters
        ----------
        remove_size_null: (bool) if True, removes instructions with size 0
        remove_adj_cap_nan: (bool) if True, removes instructions of which the 
            column adj_cap is NaN

        Example
        -------
        You adjust the capital and you store the adjustment in a column adj_cap
        however, if the adjustment is NaN, which could mean we did not find the capital,
        we do not want to include the instruction, so we remove them when looking at 
        metrics. 
        More info in the branch trading_metrics sgmtradingcore.analytics.capital_allocation.strategy
        
        Notes
        -----
        It does NOT modify the DataFrame self.concat_instr_df inplace

        Returns
        -------
        (DataFrame) filtered instructions based on the parameters
        """
        if any([remove_size_null, remove_adj_cap_nan]):  # will update instr_df
            instr_df = self.concat_instr_df.copy()
        else:  # nothing will be changed in self.concat_instr_df so no need to copy
            instr_df = self.concat_instr_df
        if instr_df.empty:
            raise Exception('concat_instr_df is empty. Cannot display metrics')
        if remove_size_null:
            no_instrs = len(instr_df)
            idx_size_null = instr_df['stake'] == 0
            if sum(idx_size_null) > 0:
                self.logger.warning('Dropping {}/{} instructions with stake 0'.format(
                    sum(idx_size_null), no_instrs))
                instr_df = instr_df.loc[~idx_size_null]
        if remove_adj_cap_nan:
            if 'adj_cap' not in instr_df.columns:
                raise Exception('concat_instr_df do not have column '
                                'adj_cap and so cannot remove_adj_cap_nan')
            no_instrs = len(instr_df)
            idx_nan = instr_df['adj_cap'].isnull()
            if sum(idx_nan) > 0:
                self.logger.warning('Keeping {}/{} instructions with known capital adjustment'
                                    .format(no_instrs - sum(idx_nan), no_instrs))
                # remove nan in adjst_capital
                instr_df = instr_df.loc[~idx_nan]
        return instr_df


class StrategyBacktestSettings(StrategySettings):
    pass