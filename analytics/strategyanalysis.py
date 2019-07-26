from collections import defaultdict
from datetime import timedelta, date, datetime
import warnings

from sgmtradingcore.analytics.metrics import get_liability, get_equivalent_odds
from sgmtradingcore.analytics.metrics import flat_capital_metrics, flat_capital_traces
from stratagemdataprocessing import data_api
from sgmtradingcore.execution.monitoring import json_to_bet_info
from stratagemdataprocessing.dbutils.mongo import MongoPersister


import numpy as np
import operator
import pandas as pd
import datetime as dt
import pytz
from bson.objectid import ObjectId
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)


class StrategyPerformance(object):
    """
    Class to assess performance of a strategy against random, producing plots, etc..
    df_bets: dataframe with at least ['date', 'pnl', 'stake', 'is_back', 'odds'] and 'capital', if you don't provide one
    """
    def __init__(self, df_bets, capital=None, commission=0., rng_seed=123456):
        self._commission = commission
        self._capital = capital

        self._real_bets = df_bets
        self._real_bets = self._preprocess_dataframe(self._real_bets, commission=self._commission)

        self._control_bets = None
        self._control_financial_metrics = None
        self._rng = np.random.RandomState(rng_seed)



        self._real_financial_metrics = flat_capital_metrics(self._real_bets, capital=self._capital).T.squeeze()
        self._real_traces = flat_capital_traces(self._real_bets, capital=self._capital)


    def _preprocess_dataframe(self, df, commission=0.025):
        # note: this assumes that the commissions are not taken out yet.
        self._check_dataframe(df)
        idx_win = df['pnl'] > 0.
        df.loc[idx_win, 'pnl'] = df.loc[idx_win, 'pnl']*(1. - commission)
        df = reshape_dataframe(df)
        return df

    def _check_dataframe(self, df):
        cols_required = ['dt', 'pnl', 'stake', 'is_back', 'odds']
        if self._capital is None:
            cols_required += ['capital']
        for el in cols_required:
            assert el in df.columns

        wrong_odds = np.sum(df['odds'] < 1.)
        if wrong_odds > 0:
            raise Exception('Your dataframe has some odds < 1.')
        return df[cols_required]

    @property
    def default_significance_metrics(self):
        return [
            'hit_ratio',
            'unitary_stake_return',
            'cum_return',
            'volatility (not annualised)',
            'sharpe_ratio',
            'maximum_drawdown',
            # 'maximum_runup',
            # 'total_pnl'
        ]

    @property
    def real_bets(self):
        return self._real_bets

    @property
    def control_bets(self):
        return self._control_bets

    @property
    def real_financial_metrics(self):
        return self._real_financial_metrics

    @property
    def control_financial_metrics(self):
        return self._control_financial_metrics

    @property
    def real_traces(self):
        return self._real_traces

    @property
    def control_traces(self):
        return self._control_traces

    @property
    def summary_randomised_metrics(self):
        if self._control_financial_metrics is not None:
            return self._control_financial_metrics.describe()
        else:
            warnings.warn('You need to calculate_randomised_metrics first')

    def create_randomised_outcomes(self, n_samples=10):
        raise Exception('Method must be implemented in subclass')


    def calculate_randomised_metrics(self):
        if self._control_bets is None:
            raise Exception('you need to create randomised outcomes first!')

        self._control_financial_metrics = self._control_bets.groupby('n_sample').apply(flat_capital_metrics,
                                                                                       capital=self._capital)

        return self._control_financial_metrics

    def calculate_randomised_traces(self):
        if self._control_bets is None:
            raise Exception('you need to create randomised outcomes first!')

        self._control_traces = self._control_bets.groupby('n_sample').apply(flat_capital_traces,
                                                                                       capital=self._capital)

        return self._control_traces

    def significance_test(self, metrics=None):
        """
        It returns the percentile of the real value wrt the random results
        :param metrics:
        :return:
        """
        if self._control_financial_metrics is None:
            raise Exception('you need to create randomised calculate_randomised_metrics first!')

        if metrics is None:
            metrics = self.default_significance_metrics

        significance = OrderedDict()
        for metric in metrics:
            significance[metric] = calculate_percentile(self._real_financial_metrics[metric], self._control_financial_metrics[metric])

        self._metrics_percentiles = pd.Series(significance)

        return self._metrics_percentiles

    def plot_significance_test(self, metrics=None):
        """
        It plots the distribution of the random metrics and the realised one
        :param metrics:
        :return:
        """
        if self._control_financial_metrics is None:
            raise Exception('you need to create randomised calculate_randomised_metrics first!')

        if metrics is None:
            metrics = self.default_significance_metrics

        metrics_to_plot = self._control_financial_metrics[metrics]

        melt_control_metrics = pd.melt(metrics_to_plot, var_name='metric', value_name='value')
        melt_control_metrics['data_type'] = 'control'

        g = sns.FacetGrid(melt_control_metrics,
                  col='metric',
                  hue = 'data_type',
                  col_wrap=2,
                  size=4, aspect=1.5,
                  sharex=False,
                  sharey=False,
                  col_order=metrics
                )

        bin_n = int(np.sqrt(len(metrics_to_plot.index)))

        g.map(sns.distplot, 'value', norm_hist=True, axlabel=False, bins=bin_n)

        axes = g.axes
        for i, col in enumerate(metrics):
            ylim = axes[i].get_ylim()
            x = self.real_financial_metrics[col]
            axes[i].plot([x, x], [0., 100.], 'r')
            axes[i].set_ylim(ylim)

        fig = plt.gcf()

        return fig

    def plot_significance_traces(self):

        if self.control_traces is None:
            raise Exception('you need to create randomised calculate_randomised_traces first!')

        plot_real_trace = self.real_traces.reset_index()
        plot_control_trace = self.control_traces.reset_index()
        plot_control_trace['control_traces'] = ''

        g = sns.FacetGrid(plot_control_trace,
                  col = 'control_traces',
                  hue = 'n_sample',
                  size=4, aspect=2,
                  sharex=False,
                  sharey=False,
                  )
        g.map(plt.plot, 'date', 'cum_return', color='g', lw=0.05)

        axes = g.axes[0,0]
        axes.plot(plot_real_trace['date'].values, plot_real_trace['cum_return'].values, lw=2, color='r')

        fig = plt.gcf()

        return fig


# ----------------- class to assess strategy degradation

class StrategyPerformanceControl(StrategyPerformance):

    def __init__(self, df_bets, capital=None, commission=0., rng_seed=123456):

        super(StrategyPerformanceControl, self).__init__(df_bets, capital=capital,
                                                         commission=commission, rng_seed=rng_seed)

    def create_randomised_outcomes(self, n_samples=10, commission=0.):
        self._control_bets = get_resampled_dataframe(self._real_bets, n_samples=n_samples,
                                                     commission=commission, rng=self._rng)
        self._n_samples = n_samples
        return self._control_bets


class StrategyPerformanceThroughTime(StrategyPerformance):
    """
    Class to assess performance of a strategy previous behaviour, producing plots, etc..
    df_bets: dataframe with at least ['date', 'pnl', 'stake', 'is_back', 'odds'] and 'capital', if you don't provide one
    """
    def __init__(self, df_bets, df_bets_control=None, capital=None, commission=0.,
                 n_testing=100, date_split=None, rng_seed=123456):

        if df_bets_control is None:
            if date_split is not None:
                warnings.warn('The n_testing parameter is going to be ignored as you selected a date_split')
                date_split = dt.datetime.strptime(date_split, '%Y-%m-%d')

                n_testing = np.sum(df_bets['dt'] > date_split)

            total_bets = len(df_bets.index)
            n_control_bets = total_bets - n_testing

            if  total_bets < n_testing:
                raise Exception('Number of total bets ({}) cannot be < n_last_trades {}'.format(total_bets, n_testing))
            elif n_control_bets < 100:
                warnings.warn('Only {} trades left as control!'.format(n_control_bets))

            print '\n N total bets: {}'.format(total_bets)
            print 'N control bets: {}'.format(n_control_bets)
            print 'N testing bets: {}'.format(n_testing)

            df_bets = df_bets.sort('dt').reset_index(drop=True)
            self._df_bets_control_tosample = df_bets.loc[:n_control_bets]
            self.date_range_control = {'start': self._df_bets_control_tosample['dt'].min(),
                                       'stop': self._df_bets_control_tosample['dt'].max()}


            df_bets_test = df_bets.loc[n_control_bets:].reset_index(drop=True)
            self.date_range_test = {'start': df_bets_test['dt'].min(),
                                    'stop': df_bets_test['dt'].max()}
        else:
            print 'You have passed a dataframe of control bets. \n ' \
                  'n_testing and date_spilt parameters are going to be ignored!'

            n_testing = len(df_bets.index)
            n_control_bets = len(df_bets_control.index)

            print 'N control bets: {}'.format(n_control_bets)
            print 'N testing bets: {}'.format(n_testing)

            self._df_bets_control_tosample = df_bets_control
            df_bets_test = df_bets.copy()

        self._n_testing = n_testing

        super(StrategyPerformanceThroughTime, self).__init__(df_bets_test, capital=capital,
                                                             commission=commission, rng_seed=rng_seed)

        self._df_bets_control_tosample = self._preprocess_dataframe(self._df_bets_control_tosample,
                                                                    commission=self._commission).reset_index(drop=True)

    def create_randomised_outcomes(self, n_samples=10):

        n_control = len(self._df_bets_control_tosample.index)
        n_to_draw = n_samples * self._n_testing

        idx = self._rng.choice(n_control, n_to_draw, replace=True)

        df_bets_control_sampled = self._df_bets_control_tosample.loc[idx].reset_index(drop=True)
        n_sample = []

        for i in np.arange(n_samples):
            n_sample += [i]*self._n_testing

        df_bets_control_sampled['n_sample'] = n_sample
        dt_list = list(self._real_bets.dt.values) * n_samples
        df_bets_control_sampled['dt'] = dt_list

        self._n_samples = n_samples

        self._control_bets = df_bets_control_sampled.copy()

        return self._control_bets

def reshape_dataframe(_df):
    df = _df.copy()

    df['liability'] = df.apply(get_liability, axis=1)
    df['equivalent_odds'] = df.apply(get_equivalent_odds, axis=1)
    df['market_probability'] = 1./ df['equivalent_odds']
    df['unitary_return'] = df['pnl'] / df['liability']
    return df


def one_resample_outcomes(_df, commission=0., rng=None):
    if rng is None:
        rng = np.random.RandomState()

    df = _df.copy()
    df['rnd'] = rng.rand(len(df.index))
    wins = df['market_probability'] > df['rnd']
    df['pnl'] = - df['liability']
    df.loc[wins, 'pnl'] = df.loc[wins, 'liability'] * (df.loc[wins, 'equivalent_odds'] - 1) * (1 - commission)
    df['unitary_return'] = df['pnl'] / df['liability']
    return df


def get_resampled_dataframe(_df, n_samples=10, commission=0., rng=None):
    out = []
    n = np.arange(1,n_samples+1)
    for j in n:
        temp = one_resample_outcomes(_df, commission=commission, rng=rng)
        temp['n_sample'] = j
        out.append(temp)
    return pd.concat(out)

def calculate_percentile(x, values):
    n_worse = np.sum(values < x)
    n_tot = len(values)

    percentile = float(n_worse) / n_tot

    return percentile



# Get real trades
def get_traded_orders(trading_user_id, strategy, strategy_descr, sport, style, start_dt, end_dt):
    settled_orders_expanded = data_api.get_settled_orders(
        trading_user_id, start_dt, end_dt, strategy, strategy_descr)

    currencies = data_api.get_currency_rates([])

    settled_orders = [
        json_to_bet_info(order_expanded, currencies=currencies) for order_expanded in settled_orders_expanded]

    try:
        capital_timeseries = data_api.get_capital_timeseries(
            trading_user_id, sport, style, start_dt, end_dt, strategy, strategy_descr=None)
    except:
        capital_timeseries = None

    def trades_to_series(x):
        out = OrderedDict()
        out['event'] = x.event
        try:
            out['instruction_id'] = ObjectId(x.instruction_id)
        except:
            out['instruction_id'] = np.nan
        out['dt'] = x.placed_dt
        out['date'] = x.placed_dt.date()
        out['is_back'] = x.is_back
        out['stake'] = x.matched_amount
        out['odds'] = x.matched_odds
        out['pnl'] = x.pnl
        return pd.Series(out)

    df_orders = pd.DataFrame([trades_to_series(x) for x in settled_orders])
    if capital_timeseries is not None:
        df_capital = pd.Series(capital_timeseries).to_frame(name='capital').reset_index().rename(columns={'index': 'date'})
        df_orders = pd.merge(df_orders, df_capital).sort('dt')
    else:
        df_orders['capital'] = 100000.
    print df_orders.columns
    df_orders = df_orders[df_orders['odds'] > 0.].reset_index(drop=True)
    df_orders['liability'] = df_orders.apply(get_liability, axis=1)

    instruction_id = list(df_orders.instruction_id.values)

    db_name = 'trading'
    coll_name = 'instructions'
    trading_mongo = MongoPersister.init_from_config(db_name, auto_connect=True)
    query = {'id': {'$in': instruction_id}}
    instructions = trading_mongo.db[coll_name].find(query)


    def instructions_to_series(x):
        out = OrderedDict()
        out['instruction_id'] = x['id']
        for key in x['details']:
            out['i_{}'.format(key)] = x['details'][key]
        return pd.Series(out)

    instructions_df = pd.DataFrame([instructions_to_series(x) for x in instructions])
    instructions_df = instructions_df.rename(columns={'id': 'order_id'})
    #df_orders = pd.merge(df_orders, instructions_df, how='left').sort('dt')

    return df_orders

if __name__ == '__main__':
    from sgmtradingcore.strategies.config.configurations import (
        PROD_CONFIG, PROD_ALGO_SPORTS_CONFIG, DEV_CONFIG, BETA_CONFIG, BETA_MM_CONFIG, BETA_MM_MATCHBOOK_CONFIG,
        PROD_BUTTERFLY_CONFIG, TRADING_USER_MAP)
    from sgmtradingcore.execution.monitoring import MonitoringEngineConnector, json_to_bet_info
    from datetime import datetime, timedelta
    import pandas as pd

    trading_user_id = PROD_ALGO_SPORTS_CONFIG['trading_user_id']
    strategy = 'tennis_sip'
    strategy_descr = 'tennis_sip_v1'
    sport = 2
    style = 'in-play'

    end_dt = datetime.now() - timedelta(days=0)
    start_dt = end_dt - timedelta(days=300)

    df_orders = get_traded_orders(trading_user_id, strategy, strategy_descr, sport, style,
                                  start_dt, end_dt)