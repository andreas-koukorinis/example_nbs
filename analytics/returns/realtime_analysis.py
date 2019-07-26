import itertools
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from IPython.core.display import display, HTML
from matplotlib import pyplot as plt

from sgmtradingcore.analytics.metrics import get_liability, financial_metrics
from sgmtradingcore.analytics.returns.backtest_analysis import \
    StrategiesBacktestAnalysis
from sgmtradingcore.analytics.returns.bar_plot_events_importance import \
    plot_boxplot, ret_on_stake, total_pnl
from sgmtradingcore.analytics.returns.util import StrategySettings
from sgmtradingcore.core.trading_types import OrderOutcome
from sgmtradingcore.strategies.config.configurations import TRADING_USER_MAP
from sgmtradingcore.strategies.strategy_base import StrategyStyle
from stratagemdataprocessing.data_api import get_capital_timeseries_all_descrs, \
    NoAllocationFoundException, get_settled_orders, get_instructions_by_id
from stratagemdataprocessing.parsing.common.stickers import parse_sticker, \
    MARKET_ABBR, SELECTION_ABBR

__author__ = 'tduriez'


class StrategyRealtimeSettings(StrategySettings):
    def __init__(self, strategy_name, strategy_desc, strategy_code,
                 trading_user_id):
        # the mnemonic should always been prod if realtime
        super(StrategyRealtimeSettings, self).__init__(
            strategy_name, strategy_desc, strategy_code,
            'prod', trading_user_id)


"""Function helper to load strategies results and display metrics and 
performance
"""



class StrategiesRealtimeAnalysis(object):
    """
    Base abstract class that help analysing returns from realtime.

    You can look at the implementation for football to know how to implement
    the abstract methods of this abstract class. The code is in
    sgmfootball/trading/realtime_analysis/base.py

    Most interesting features use of the class include:
        - Loading all the results from all the strategy descriptions from a
            strategy using the method add_strategy
        - Make the capital constant per strategy description using the method
            make_capital_constant
        - Accessing all the trades from the added strategies in a DataFrame
            using the property trades
        - Analysing trades assuming 100% fill rates using property
            trades_filled and compare realised and 100 filled using the
            property trades_realised_and_filled
        - Play with the theorical capital by using display_capital_summary
            and setting capital as the overall capital of as the sport capital
            of one capital per strategy name or one capital per strategy desc
            using the methods set_capital_as_xxx
        - Display the financial metrics using display_financial_metrics
        - Display basic returns pnl and stake stat using display_stake_return
        - Display box plots from leaving out events using plot_pnl_boxplot
            and plot_ret_stake_boxplot



    TODO generalise to orders (currently only working with instructions)
    """
    # TODO make it possible to load strategies from different users
    # but is a bit difficult to handle everywhere (e.g. with capital alloc)

    def __init__(self, start_dt, end_dt, trading_user_id, sport):
        # parameters common to all the strategies to analyse
        self._start_dt = start_dt
        self._end_dt = end_dt
        self._trading_user_id = trading_user_id
        self._sport = sport
        self._weights_algo = {}  # strategy

        self._strategy_styles = {}  # strategy -> StrategyStyle
        self._strategy_capitals = {}  # strategy -> DataFrame

        self._settings_trades = {}  # StrategySettings -> DataFrame
        self._settings_orders = {}  # StrategySettings -> DataFrame

        self._is_banned = {}  # strategy_desc -> bool

        self._trades = None  # helper for the property trades
        self._trades_filled = None  # helper for the property trades_filled
        self._trades_realised_and_filled = None  # helper for the property
        self._orders = None  # helper for the property orders
        self._capitals = None  # helper for the property allocations

        # parameters for adjusting the capital to make it constant
        self._constant_capital = None  # helper for the property
        self._capital_adjustment = None  # to record how we've changed the capital
        self._size_placed_col = 'size_placed'
        assert self._size_placed_col in self._columns_capital_adjusted()

    @property
    def trades(self):
        """
        Returns
        -------
        - DataFrame, with columns "date", "fixture_index", "event_id", "dt"
            contatenation of all the trades added to the instance so far.
        """
        if self._trades is None:
            all_trades = []
            for settings_key, trades in self._settings_trades.iteritems():
                trades = trades.copy()
                settings = StrategyRealtimeSettings.init_from_key(settings_key)
                # add to the instructions where it comes from
                for k, v in settings.__dict__.iteritems():
                    trades[k] = v
                if self._capital_adjustment is not None:
                    self._adjust_capital(trades, settings.strategy_desc)
                    trades = trades.loc[~trades['capital'].isnull()]
                all_trades.append(trades)
            if len(all_trades) == 0 or all([t is None for t in trades]):
                raise Exception('Please add trades using add_strategy before'
                                ' calling this property. No trades found')

            trades = pd.concat(all_trades, axis=0)

            trades['date'] = trades['dt'].apply(lambda x: x.date())
            trades = trades.sort_values(by=['date','event_id'])
            # WARNING : this assumes that all the instructions are placed the
            # same day.
            trades['fixture_index'] = trades['event_id'].ne(
                trades['event_id'].shift()).cumsum()
            trades = self._add_useful_columns(trades)
            trades = self._add_id(trades)
            self._trades = trades
        return self._trades

    @property
    def trades_filled(self):
        """
        Returns
        -------
        - DataFrame, with columns "date", "fixture_index", "event_id", "dt"
            contatenation of all the trades added to the instance so far with
            trades assumed to have been filled 100% instead of what happened
        """
        if self._trades_filled is None:
            if 'fill_rate' not in self.trades.columns:
                raise Exception('Please add column fill_rate in the trades '
                                '(inf method _add_useful_info method')
            trades_filled = self.trades.copy()
            unknown_columns = []
            for c in trades_filled.columns:
                if c == self._size_placed_col:
                    # do not modify initial intention size !
                    continue
                if c in self._columns_capital_adjusted():
                    trades_filled.loc[:, c] /= trades_filled['fill_rate']
                elif c not in self._columns_not_capital_adjusted():
                    unknown_columns.append(c)
                else:
                    pass  # column not affected by size
            if len(unknown_columns) > 0:
                logging.warning(
                    'Columns {} might be wrong after changing the size'
                    ' while filling trades if it is size dependend. Please add'
                    'the columns to either _columns_not_size_dependent or '
                    '_columns_size_dependent'.format(unknown_columns))
            self._trades_filled = trades_filled
        return self._trades_filled

    @staticmethod
    def is_low_pnl_strategy_desc(strategy_desc):
        """Function to tell us whether we should display the pnl and return
        on stake barplot in a different plot

        Parameters
        ----------
        strategy_desc: str

        Returns
        -------
        bool
        """
        raise NotImplementedError('To implement in subclass')

    def set_capital_as_portfolio_capital(self):
        display(HTML('<p>Setting trades capital as <b>portfolio capital.</b>'
                     'This means that the cumulative returns will be with '
                     'respect to the overall capital allocated to the '
                     'portfolio.</p>'))
        capital_adjusted = self._capital_adjustment is not None
        total_capital = self._get_portfolio_capital(
            add_adjustment=capital_adjusted)

        self.trades['capital'] = self.trades['dt'].apply(
            lambda x: self._get_instr_capital(x, total_capital))
        self.trades_filled['capital'] = self.trades_filled['dt'].apply(
            lambda x: self._get_instr_capital(x, total_capital))

    def set_capital_as_strategy_capital(self):
        display(HTML('<p>Setting trades capital as <b>strategy capital.</b>'
                     'This means that the cumulative returns will be with '
                     'respect to the overall capital allocated to the '
                     'strategy.</p>'))
        capital_adjusted = self._capital_adjustment is not None
        valid_capitals = self._get_valid_capitals(
            add_adjustment=capital_adjusted)
        total_capital = valid_capitals.sum(level='strategy')
        self.trades['capital'] = self.trades.apply(
            lambda x: self._get_instr_capital(x['dt'], total_capital.loc[
                x['strategy_name']]),
            axis=1)

        self.trades_filled['capital'] = self.trades_filled.apply(
            lambda x: self._get_instr_capital(x['dt'], total_capital.loc[
                x['strategy_name']]),
            axis=1)

    def set_capital_as_strategy_desc(self):
        display(HTML('<p>Setting trades capital as <b>strategy description '
                     'capital</b> This means that the cumulative returns will '
                     'be with respect to the capital allocated to the '
                     'strategy description.</p>'))

        capital_adjusted = self._capital_adjustment is not None
        capitals = self._get_valid_capitals(add_adjustment=capital_adjusted)
        self.trades['capital'] = self.trades.apply(
            lambda x: self._get_instr_capital(x['dt'], capitals.reset_index(
                'strategy', drop=True).loc[x['strategy_desc']]),
            axis=1)

        self.trades_filled['capital'] = self.trades_filled.apply(
            lambda x: self._get_instr_capital(x['dt'], capitals.reset_index(
                'strategy', drop=True).loc[x['strategy_desc']]),
            axis=1)

    def display_financial_metrics(self, groupby_settings, x_axis=None):
        """
        Parameters
        ----------
        groupby_settings: list,
            e.g. [(['strategy_name', 'strategy_desc'], [1, 0, None])]
        x_axis: list or str, if None, will be set to ['fixture_index' 'date']
            plot cumulative according to each value in x_axis
        """
        if x_axis is None:
            x_axis = ['fixture_index', 'date']
        for grp_by, row_hue_col in groupby_settings:
            display(HTML('<h3> Grouped by {}</h3>'.format(', '.join(grp_by))))
            display(HTML('<h4> Realised </h4>'))
            if 'date' in grp_by:
                metrics_to_show_ = ['n_trades', 'n_events', 'cr_event',
                                    'hit_ratio', 'cr_trade', 'cum_return',
                                    'total_pnl', 'unitary_stake_return']
            else:
                metrics_to_show_ = self.metrics_to_show()
            rename_metrics = self.rename_metrics()

            display(financial_metrics(self.trades, 'date',
                                      groupby=grp_by)[
                        metrics_to_show_].rename(columns=rename_metrics))
            display(HTML('<h4> Assuming 100% fill rate </h4>'))
            display(financial_metrics(self.trades_filled, 'date',
                                      groupby=grp_by)[metrics_to_show_].rename(
                columns=rename_metrics))
            for x_label in x_axis:
                StrategiesBacktestAnalysis.static_display_trace(
                    self.trades, grp_by, x_label, row_hue_col=row_hue_col)

    def display_stake_return(self, groupby):
        trades_aggr = self.trades.groupby(groupby).sum()[
            ['pnl', 'stake']].reset_index()
        display(HTML('unit stake return <b>{}</b> from {} {}'.format(
            (trades_aggr['pnl'] / trades_aggr['stake']).mean(),
            len(trades_aggr), ', '.join(groupby))))

    @staticmethod
    def metrics_to_show():
        return['n_trades', 'n_events', 'event_sharpe_ratio', 'cr_event',
               'event_sortino', 'sharpe_ratio', 'hit_ratio', 'cr_trade',
               'cr_day', 'cum_return', 'total_pnl', 'maximum_drawdown',
               'maximum_runup', 'unitary_stake_return',
               'volatility (not annualised)']

    @staticmethod
    def rename_metrics():
        return {'maximum_drawdown': 'max_DD',
                'maximum_runup': 'max_runup',
                'unitary_stake_return': 'unit_stake_return',
                'volatility (not annualised)': 'vol_not_ann',
                'volatility_annualised': 'vol_annualised'}

    @staticmethod
    def _get_instr_capital(dt, total_capital):
        before_dt = total_capital.loc[:dt.date()]
        if len(before_dt) == 0:
            return np.nan
        return before_dt.iloc[-1]

    @property
    def trades_realised_and_filled(self):
        if self._trades_realised_and_filled is None:
            trades = self.trades.copy()
            trades['fill_rate_type'] = 'Realised'
            trades_filled = self.trades_filled.copy()
            trades_filled['fill_rate_type'] = 'Assumed 100%'
            self._trades_realised_and_filled = pd.concat(
                [trades, trades_filled], axis=0)
        return self._trades_realised_and_filled

    @staticmethod
    def _columns_capital_adjusted():
        return ['pnl', 'size_matched', 'size_placed', 'stake', 'size',
                'liability', 'capital']

    @staticmethod
    def _columns_not_capital_adjusted():
        return ['dt', 'kick_off', 'is_back', 'status', 'instruction_id',
                'trade_id', 'sticker', 'selection', 'comp', 'placed_dt',
                'handicap', 'market', 'fixture_id', 'hc', 'mkt_id', 'sel',
                'strategy_desc', 'mnemonic', 'strategy_name', 'strategy_code',
                'trading_user_id', 'adjustment', 'version', 'domain', 'date',
                'id', 'fill_rate', 'av_odds', 'odds', 'odds_matched',
                'average_price_matched', 'event_id', 'market_id',
                'selection_id', 'fixture_index']

    @property
    def orders(self):
        if self._orders is None:
            raise NotImplementedError('If you want to work with orders, you '
                                      'need to write up this bit')
        return self._orders

    @property
    def capitals(self):
        if self._capitals is None:
            capitals = self._strategy_capitals.values()
            if len(capitals) == 0 or all([s is None for s in capitals]):
                raise Exception('Please add strategy using add_strategy before'
                                ' calling this property. No trades found')
            capitals = pd.concat(capitals, axis=0)
            self._capitals = capitals
        return self._capitals

    def add_strategy(self, strategy, style):
        """Add trades and orders from all the sub-strategies of strategy

        Parameters
        ----------
        strategy: str,
            strategy name of the trading strategies to add
        style: str, either 'in-play' or 'dead-ball'
            the style of the strategy.

        Returns
        -------
        Nothing, the trades and orders are added to the instance attributes
        """
        if style not in ['in-play', 'dead-ball']:
            raise ValueError('style {} not valid. Should be either dead-ball'
                             ' or in-play'.format(style))
        self._reset_properties()

        display(HTML('<p>Adding strategy <b>' + strategy + '</b></p>'))

        self._strategy_styles[strategy] = style

        capitals = self._get_strategy_capitals(strategy, style)

        self._strategy_capitals[strategy] = capitals

        realtime_credentials = get_all_strategy_realtime_settings(
            self._start_dt, self._end_dt, self._sport, strategy, style=style,
            trading_user_id=self._trading_user_id, positive_capital_only=False)

        realtime_credentials = sorted(realtime_credentials,
                                      key=lambda x: x.key)

        self._add_trades_from_credentials(realtime_credentials)

    def display_capital_summary(self, with_adjustment=False):
        total = self.capitals.reset_index('strategy', drop=True).sum()
        total_valid = self._get_portfolio_capital()

        diff_total = total - total_valid
        banned_strategies_capital = diff_total > 0
        if banned_strategies_capital.sum() > 0:
            display(HTML('<p><b>Warning<b> ! Some strategy desc banned and '
                         'had this capital allocated:'))
            display(diff_total.T.loc[banned_strategies_capital])

        if with_adjustment:
            total_valid_adjusted = self._get_portfolio_capital(
                add_adjustment=with_adjustment)
            self._display_capital_timeseries('Portfolio', total_valid,
                                             total_valid_adjusted)
        else:
            self._display_capital_timeseries('Portfolio', total_valid, None)

        strategies_capital = self._get_valid_capitals(
            add_adjustment=with_adjustment)
        strategies_capital = strategies_capital.sum(level='strategy')
        self._display_capital_timeseries('Strategies', strategies_capital,
                                         None)

    def plot_pnl_boxplot(self, groupby, frac=0.9, n=100, xlim=None,
                 separate_trading_users=True, separate_year_month=False):
        """Wrapper around plot_boxplot to display pnl barplot

        Look at plot_boxplot for further details
        """
        trades = self.trades  # change this is you want to display other trades
        return plot_boxplot(trades, total_pnl, groupby,
                            self.is_low_pnl_strategy_desc,
                            frac=frac, n=n,
                            xlim=xlim,
                            separate_trading_users=separate_trading_users,
                            separate_year_month=separate_year_month)

    def plot_ret_stake_boxplot(self, groupby, frac=0.9, n=100, xlim=None,
                        separate_trading_users=True, separate_year_month=False):
        """Wrapper around plot_boxplot to display return on stake in barplot

        Look at plot_boxplot for further details
        """
        trades = self.trades  # change this is you want to display other trades
        return plot_boxplot(trades, ret_on_stake, groupby,
                            self.is_low_pnl_strategy_desc,
                            frac=frac, n=n,
                            xlim=xlim,
                            separate_trading_users=separate_trading_users,
                            separate_year_month=separate_year_month)

    def make_capital_constant(self, capital='mean'):
        """Add information to make capital constant

        Parameters
        ----------
        capital: str, key words, either "mean", "last" or "first"
            - "mean": make the capital target the mean of the allocation period
            - "last": --------------------------- last allocated capital
            - "first: --------------------------- first allocated capital
        """
        if self._capital_adjustment is None:
            self._capital_adjustment = {}
        capitals = self.capitals.reset_index('strategy', drop=True)
        for strategy_desc, capital_ts in capitals.iterrows():
            if strategy_desc == 'total':
                continue
            if strategy_desc not in self._is_banned:
                logging.warning(
                    'strategy description {} not found in _is_banned attribute.'
                    ' Probably due to insconsistency in capital allocation DB.'
                    ' Will set it as banned'.format(strategy_desc))
                self._is_banned[strategy_desc] = True
                continue

            if self._is_banned[strategy_desc]:
                continue
            capital_target = self._get_capital_target(capital, capital_ts)
            adjustment = capital_target / capital_ts
            adjustment.loc[adjustment.isin([-np.inf, np.inf])] = np.nan

            # ffill one value as the granularity is date so the capital could
            # have changed to 0 during the day
            adjustment = adjustment.fillna(method='ffill', limit=1)
            self._capital_adjustment[strategy_desc] = adjustment
        self._reset_properties()


    def _adjust_capital(self, trades, strategy_desc):
        adjustment_timeseries = self._capital_adjustment[strategy_desc]
        adjustment_timeseries = adjustment_timeseries.sort_index()

        trades.loc[:, 'adjustment'] = trades.loc[:, 'dt'].apply(
            self._get_instr_capital, total_capital=adjustment_timeseries)
        idx_nan = trades['adjustment'].isnull()
        if sum(idx_nan) > 0:
            logging.warning('[{}] {}/{} instructions with unknown capital.'
                            'Some NaN will be found in instructions'
                            .format(strategy_desc, sum(idx_nan), len(trades)))

        unknown_columns = []
        for c in trades.columns:
            if c in self._columns_capital_adjusted():
                trades.loc[:, c] *= trades.loc[:, 'adjustment']
            elif c not in self._columns_not_capital_adjusted():
                unknown_columns.append(c)
            else:
                pass  # column not affected by size
        if len(unknown_columns) > 0:
            logging.warning('Columns {} might be wrong after changing the size'
                            ' from the capital adjustment if it is size depend'
                            'ent. Please add the columns to either _columns_'
                            'not_size_dependent or _columns_size_dependent'.
                            format(unknown_columns))

    def _get_portfolio_capital(self, add_adjustment=False):
        valid_capitals = self._get_valid_capitals(add_adjustment)
        portfolio_capital = valid_capitals.sum()
        return portfolio_capital

    def _get_valid_capitals(self, add_adjustment):
        if add_adjustment and self._capital_adjustment is None:
            raise Exception('make_capital_constant should be run before '
                            'to get adjustment in _get_portfolio_capital')
        valid_capitals = self._keep_not_banned_strategy_desc()
        if add_adjustment:
            for (strategy, desc), capital_ts in valid_capitals.iterrows():
                if self._is_banned[desc]:
                    continue
                adj_capital = self._capital_adjustment[desc].loc[
                    capital_ts.index].fillna(0)
                adj_strategy_desc = capital_ts.multiply(adj_capital)
                valid_capitals.loc[(strategy, desc)] = adj_strategy_desc
        return valid_capitals

    def _keep_not_banned_strategy_desc(self):
        capitals = self.capitals
        idx_not_banned = [not self._is_banned[sd] for sd in
                          capitals.index.get_level_values(
                              'strategy_desc')]
        capitals = capitals.iloc[idx_not_banned]
        return capitals


    def _get_capital_target(self, capital, capital_ts):
        if isinstance(capital, basestring):
            capital_ts = capital_ts.loc[~capital_ts.isnull()]
            if capital not in ['mean', 'last', 'first']:
                raise ValueError('capital should be either the amount or a key'
                                 'word mean, last or first. (given ' + capital)

            if capital == 'mean':
                capital_target = capital_ts.mean()
            elif capital == 'last':
                capital_target = capital_ts.sort_index().iloc[-1]
            elif capital == 'first':
                capital_target = capital_ts.sort_index().iloc[0]
            else:
                raise ValueError('capital keyword ' + capital + ' not valid')
        else:
            raise ValueError('Receive wrong capital {} (type {})'.format(
                capital, type(capital)))

        return capital_target

    def _display_capital_timeseries(self, label, capital,
                                    capital_adjusted=None):
        ax = capital.T.plot(label=label, figsize=(16, 7))
        plt.title(label + ' capital allocation per strategy over time')
        plt.ylim((0, capital.max().max() * 1.1))
        if capital_adjusted is not None:
            capital_adjusted.T.plot(ax=ax, label=label + ' adjusted')
        plt.legend()
        plt.show()

    def _add_id(self, trades):
        trades['id'] = trades['strategy_desc'].apply(
            self._id_from_strategy_desc)
        return trades

    def _add_useful_columns(self, trades):
        trades['fill_rate'] = trades['stake'] / trades['size_placed']
        return trades # to override if you want to add columns to all your trades

    def _reset_properties(self):
        self._trades = None
        self._trades_filled = None
        self._trades_realised_and_filled = None
        self._orders = None

    def _add_trades_from_credentials(self, realtime_credentials):
        display_str = '<p>Adding instructions and orders <ul>'
        for credential in realtime_credentials:
            if self.is_banned_strategy_desc(credential.strategy_desc):
                self._is_banned[credential.strategy_desc] = True
                display_str += '<li>BANNED - {}</li>'.format(credential)
                continue
            else:
                self._is_banned[credential.strategy_desc] = False

            trades, orders_only = self._get_realtime_orders_instructions(
                credential.strategy_name,
                credential.strategy_desc,
                credential.strategy_code,
                self._start_dt,
                self._end_dt,
                credential.trading_user_id)

            if trades is None and orders_only is None:
                display_str += '<li>NO TRADES - {}</li>'.format(credential)
                continue

            # Store trades in attributes of the class
            if credential.key in self._settings_trades:
                warnings.warn('[{}] Trades already added. Will be overridden'
                              .format(credential))
            self._settings_trades[credential.key] = trades

            # store orders in attributes of the class
            if credential.key in self._settings_orders:
                warnings.warn('[{}] Orders already added. Will be overridden'
                              .format(credential))
            self._settings_orders[credential.key] = orders_only
            display_str += '<li>ADDED - {}</li>'.format(credential)

        display_str += '</ul></p>'
        display(HTML(display_str))

    def _get_realtime_orders_instructions(self, strategy_name, strategy_desc,
                                          strategy_code,
                                          start_dt, end_dt,
                                          trading_user_id,
                                          groupby_cols='instruction_id'):
        """
        Wrapper above Mongo strategy helper

        Parameters
        ----------
        groupby_cols: The columns by which one will join instruction and orders
            to infer instructions PnL from orders

        Returns
        -------
        - orders: DataFrame
        - instructions: DataFrame
        """
        # get orders
        if isinstance(start_dt, basestring):
            start_dt = datetime.strptime(start_dt, '%Y-%m-%d')
        if isinstance(end_dt, basestring):
            end_dt = datetime.strptime(end_dt, '%Y-%m-%d')

        orders = self._get_realtime_orders(strategy_name, strategy_desc,
                                           strategy_code, trading_user_id,
                                           end_dt, start_dt)
        if orders is None:
            return None, None  # there will be no instructions neither

        unique_instr_ids = [t for t in orders.instruction_id.unique().tolist()
                            if t != -1]

        instr_df = self._get_realtime_instructions(strategy_name,
                                                   strategy_desc,
                                                   strategy_code,
                                                   unique_instr_ids,
                                                   trading_user_id)

        instr_df = self._add_instructions_pnl_from_orders(instr_df, orders,
                                                          groupby_cols)
        return instr_df, orders

    def _add_instructions_pnl_from_orders(self, instr_df, orders,
                                          groupby_cols):
        # Control the view of pnl by grouping by
        if groupby_cols is not None:
            orders_pnl = orders.groupby(groupby_cols)[['pnl']].sum()
            instr_df = pd.merge(instr_df, orders_pnl, left_on=groupby_cols,
                                right_index=True, how='inner')
        return instr_df

    def _get_realtime_instructions(self, strategy_name, strategy_desc,
                                   strategy_code, unique_instr_ids,
                                   trading_user_id):
        settled_instructions = get_instructions_by_id(trading_user_id,
                                                      unique_instr_ids)
        instr_df = pd.DataFrame(
            map(lambda x: self._format_instruction(
                x, strategy_name, strategy_desc, strategy_code),
                settled_instructions))
        if not instr_df.empty:
            # Get rid of void instructions
            instr_df = instr_df[instr_df['odds'] > 0.]
            instr_df.dt = pd.to_datetime(instr_df.dt)
        return instr_df

    def _get_realtime_orders(self, strategy_name, strategy_desc, strategy_code,
                             trading_user_id, end_dt, start_dt):
        orders = get_settled_orders(trading_user_id, start_dt, end_dt,
                                    strategy=strategy_name,
                                    strategy_descr=strategy_desc)
        orders = pd.DataFrame(
            map(lambda x: self._format_order(
                x, strategy_name, strategy_desc, strategy_code),
                orders))
        if orders.empty:
            return None
        orders = orders.loc[~orders['trade_id'].isnull()]
        orders = orders[orders.odds > 0].reset_index(drop=True)
        if orders.empty:
            return None
        orders['placed_time'] = pd.to_datetime(orders.placed_time)
        orders['date'] = orders.placed_time.apply(lambda tt: tt.date())
        orders['liability'] = orders.apply(
            lambda row: get_liability(row, stake_col='stake',
                                      odds_col='odds'), axis=1)
        return orders

    def _format_order(self, order, strategy_name, strategy_desc,
                      strategy_code):
        formatted_order = self._format_order_base(order)
        formatted_order.update(self._format_order_strategy(
            order, strategy_name, strategy_desc, strategy_code))
        return formatted_order

    def _format_order_strategy(self, order, strategy_name, strategy_desc,
                               strategy_code):
        """
        Should be overridden in subclass sport specific to add some information
        according to the information contained in the order. It will usually
        depend on the strategy and strategy versions so it cannot be general

        Parameters
        ----------
        order: Order
        strategy_name: str,
        strategy_desc: str,
        strategy_code: str,

        Returns
        -------
        dict
        """
        return {}

    def _format_order_base(self, order):
        """
        Better to keep this base common. You can add things by overriding and
        calling the super method first and add things

        Parameters
        ----------
        order: (Order)

        Returns
        -------
        dict
        """

        fx = order['exchange_rate']
        fx = 1. * fx if fx > 0 else 1.
        formatted_order = {
            'status': order['status'],
            'instruction_id': order.get('instruction_id', -1),
            'odds': order['average_price_matched'],
            'size_placed': order['size'] / fx,
            'stake': order['size_matched'] / fx,
            'size_lapsed': order.get('size_lapsed', 0.) / fx,
            'size_rejected': order['size_rejected'] / fx,
            'size_cancelled': order.get('size_cancelled', 0.) / fx,
            'placed_time': order.get('placed_time', None),
            'trade_id': order.get('trade_id', None),
            # 'dt': order['ut'],
            'is_back': order['bet_side'] == 'back'
        }

        self._add_sticker_info(order['sticker'], order, formatted_order)

        if 'outcome' in order:
            formatted_order.update({
                'gross_pnl': order['outcome']['gross'] / fx,
                'pnl': order['outcome']['net'] / fx,
                'commission': order['outcome']['commission'] / fx,
                'outcome': OrderOutcome(order['outcome']['id'])
            })
        else:
            formatted_order.update({
                'gross_pnl': 0.,
                'pnl': 0.,
                'commision': 0.,
                'outcome': OrderOutcome.UNKNOWN,
            })
        return formatted_order

    def _add_sticker_info(self, sticker, order, info):
        sport, (market_scope, gsm_id), market_id, params, bm = parse_sticker(
            sticker)
        if sport != self._sport:
            raise Exception('Received a sticker which is not from the correct'
                            ' sport {} ! {}'.format(self._sport, order))
        sticker_info = {
            'sticker': sticker,
            'event_id': gsm_id,
            'fixture_id': int(gsm_id[3:]),
            'handicap': None if len(params) == 1 else params[1],
            'selection_id': params[0],
            'selection': SELECTION_ABBR[self._sport][params[0]],
            'market': MARKET_ABBR[self._sport][market_id],
            'market_id': market_id}
        info.update(sticker_info)

    def _format_instruction(self, instruction, strategy_name, strategy_desc,
                            strategy_code):
        formatted_instr = self._format_instruction_base(instruction)
        formatted_instr.update(self._format_instruction_strategy(
            instruction, strategy_name, strategy_desc, strategy_code))
        return formatted_instr

    def _format_instruction_base(self, instruction):
        """
        Better to keep this base common. You can add things by overriding and
        calling the super method first and add things

        Parameters
        ----------
        instruction: (Instruction)

        Returns
        -------
        dict
        """

        formatted_instruction = {
            'instruction_id': instruction['id'],
            'trade_id': instruction['trade_id'],
            'odds': instruction['average_price_matched'],
            'stake': instruction['size_matched'],
            'size_placed': instruction['size'],
            'dt': instruction['placed_time'],
            'is_back': instruction['bet_side'] == 'back',
            'capital': instruction['details']['capital'],
            'status': instruction['status']
        }
        self._add_sticker_info(instruction['sticker'], instruction,
                               formatted_instruction)

        return formatted_instruction

    def _format_instruction_strategy(self, instruction, strategy_name,
                                     strategy_desc, strategy_code):
        """
        Should be overridden in subclass sport specific to add some information
        according to the information contained in the instruction. It will
        usually depend on the strategy and strategy versions so it cannot be
        general

        Parameters
        ----------
        instruction: Instruction,
        strategy_name: str,
        strategy_desc: str,
        strategy_code: str,

        Returns
        -------
        dict
        """

        return {}

    def _get_strategy_capitals(self, strategy, style):
        capitals = get_capital_timeseries_all_descrs(
            self._trading_user_id, self._sport, style,
            self._start_dt, self._end_dt, strategy,
            keep_strategy_alloc=False, convert_to_date=True)
        capitals = pd.DataFrame(capitals)
        capitals.index.name = 'strategy_desc'
        capitals['strategy'] = strategy
        capitals = capitals.set_index('strategy', append=True)
        capitals = capitals.reorder_levels(order=['strategy', 'strategy_desc'])
        return capitals

    def is_banned_strategy_desc(self, strategy_desc):
        """
        To override if you want to exclude some strategy decsriptions
        """
        return False

    @staticmethod
    def _id_from_strategy_desc(strategy_desc):
        """
        To override if you want to see two strategy desc as one.
        For example, if strategy_v0 and strategy_v1 are the same strategies
        but with just a change in sizing, maybe you want to see it as the
        same strategy v0_1

        Parameters
        ----------
        strategy_desc: str, strategy description of a strategy

        Returns
        -------
        id: str
        """
        return strategy_desc


def get_all_strategy_realtime_settings(start_dt, end_dt, sport, strategy_name,
                                       style=None, trading_user_id=None,
                                       positive_capital_only=True):
    """
    Warnings
    --------
    Will provide all the StrategyRealTimeSetting with strategy_code None as it 
    cannot know it for now with just the capital, it will take more work to get
    the strategy_code, probably by reading the orders placed etc..
    
    Parameters
    ----------
    start_dt: (dt)
    end_date: (dt)
    sport: (Sports) enum
    strategy_name: (str)
    style: (str, or list of str or None)
        if None, it will get all the styles
    trading_user_id: (str or list of str or None) 
        if None, it will get all the settings for the 3 accounts (dev,
         stratagem and algo)
        
    Returns
    -------
    realtime_strategy_settings : (list of StrategyRealtimeSettings)
    """

    if trading_user_id is None:
        trading_user_id = TRADING_USER_MAP.keys()
    if isinstance(trading_user_id, str):
        trading_user_id = [trading_user_id]

    if style is None:
        style = [StrategyStyle.to_str(StrategyStyle.DEADBALL),
                 StrategyStyle.to_str(StrategyStyle.INPLAY),
                 StrategyStyle.to_str(StrategyStyle.INVESTMENT)]
    if isinstance(style, str):
        style = [style]

    realtime_strategy_settings = []
    seen_already = []
    for trading_uid, style in itertools.product(*[trading_user_id, style]):
        try:
            capitals = get_capital_timeseries_all_descrs(
                trading_uid, sport, style, start_dt, end_dt, strategy_name,
                keep_strategy_alloc=False)
        except NoAllocationFoundException:
            warnings.warn('Could not get capital timeseries for trading user'
                          '{} and style {}'.format(trading_uid, style))
            continue

        for d_info in capitals.itervalues():
            for strategy_desc, capital in d_info.iteritems():
                if capital > 0 or not positive_capital_only:
                    srs = StrategyRealtimeSettings(strategy_name,
                                                   strategy_desc,
                                                   None, trading_uid)
                    if srs.key in seen_already:
                        continue
                    else:
                        realtime_strategy_settings.append(srs)
                        seen_already.append(srs.key)
    return realtime_strategy_settings

