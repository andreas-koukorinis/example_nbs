import itertools
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sgmtradingcore.strategies.config.configurations import TRADING_USER_MAP

__author__ = 'ioanna, tduriez'

"""Plot the distribution of a value from subsets of the events to measure the
impact of having a 
"""

def plot_boxplot(trades, func, groupby, is_low_value_strategy_desc,
                 frac=0.9, n=100, xlim=None,
                 separate_trading_users=True,
                 separate_year_month=False):
    """Plot the distribution of the value returned by a function func, from
    n simulations which include a fraction frac of the fixtures. This aims to
    measure how much a small subset of the event affect the value ! (e.g. pnl)

    Parameters
    ----------
    trades: DataFrame, one row per trade
    func: function which extracts a single value to plot from a row in trades
    groupby: list of str,
        how to groupby in plot, there will be one bar per group in the plot
    is_low_value_strategy_desc: function str -> bool,
        which takes a strategy description name and tell us whether it is a
        low value (returned by func) strategy to plot them in a different
        graph
    frac: float, between 0 and 1
        fraction of events to get for each simulation
    n: int,
        number of simulations to perform
    xlim: tuple (xlim min, xlim max)
    separate_trading_users: bool
    separate_year_month: bool

    Returns
    -------
    Nothing, it plots the bar plots
    """
    if not 0 <= frac <= 1:
        raise ValueError('frac should be between 0 and 1 (given {})'.format(
            frac))

    high_level_categories = []
    if separate_year_month:
        if 'year_month' in trades.columns:
            high_level_categories.append('year_month')
        else:
            warnings.warn('separate_year_month is True but column '
                          'year_month not found in trades. Ignored')
    if separate_trading_users:
        if 'trading_user_id' in trades.columns:
            high_level_categories.append('trading_user_id')
        else:
            warnings.warn('separate_trading_users is True but column '
                          'trading_user_id not found in trades. Ignored')

    plot_group_by = [nm for nm in groupby if nm not in high_level_categories]

    if 'year_month' in high_level_categories:
        for user_id, mon in itertools.product(
                trades['trading_user_id'].unique(),
                trades['year_month'].unique()):
            title = '{}_{}'.format(TRADING_USER_MAP[user_id], mon)

            df = trades[(trades['trading_user_id'] == user_id) &
                        (trades['year_month'] == mon)]

            plot_two_boxplots_xlim(df, func, plot_group_by,
                                   is_low_value_strategy_desc,
                                   frac, n, xlim, title)

    else:
        if 'trading_user_id' in high_level_categories:
            for user_id in trades['trading_user_id'].unique():
                title = '{}'.format(TRADING_USER_MAP[user_id])

                df = trades[(trades['trading_user_id'] == user_id)]

                plot_two_boxplots_xlim(df, func, plot_group_by,
                                       is_low_value_strategy_desc, frac, n, xlim,
                                       title)
        else:
            title = 'All trading users'
            plot_two_boxplots_xlim(trades, func, plot_group_by,
                                   is_low_value_strategy_desc, frac, n,
                                   xlim, title)


def plot_two_boxplots_xlim(df, func, group_by, is_low_pnl_strategy_desc,
                           frac, n, xlim, title):
    low_pnl_names = [str_desc for str_desc in df.strategy_desc.unique()
                     if is_low_pnl_strategy_desc(str_desc)]
    low_df = df[df.strategy_desc.isin(low_pnl_names)]
    new_xlim = [xl / 5.0 for xl in xlim] if xlim is not None else None

    plot_single_boxplot(low_df, func, group_by, frac, n, new_xlim, title)

    high_pnl_names = [str_desc for str_desc in df.strategy_desc.unique()
                      if not is_low_pnl_strategy_desc(str_desc)]

    high_df = df[df.strategy_desc.isin(high_pnl_names)]
    plot_single_boxplot(high_df, func, group_by, frac, n, xlim, title)


def get_bootstrapped_metric(df, func, frac, n):
    unique_event_ids = df.event_id.unique()

    size = int(np.floor(frac * len(unique_event_ids)))

    events_lists = [np.random.choice(unique_event_ids, size=size,
                                     replace=False)
                    for _ in range(n)]

    return pd.DataFrame([func(df[df.event_id.isin(events_lists[i])])
                         for i in range(n)])


def plot_single_boxplot(df, func, groupby, frac, n, xlim, title):
    grouped_pnls = df.groupby(groupby).apply(
        lambda xx: get_bootstrapped_metric(xx, func, frac, n))

    grouped_pnls = grouped_pnls.unstack(level=len(groupby)).T

    if len(grouped_pnls) > 0:
        f, ax = plt.subplots(figsize=(10, 0.7 * grouped_pnls.shape[1]))
        sns.boxplot(data=grouped_pnls, orient="h", whis=[5, 95])
        _ = plt.title(title)
        _ = plt.xlabel(func.__name__)
        if xlim is not None:
            _ = plt.xlim(xlim)

        # Calculate number of obs per group & median to position labels
        medians = grouped_pnls.mean().values + grouped_pnls.std().values
        nobs = df.groupby(groupby).event_id.apply(lambda x: len(np.unique(x)))
        nobs = [str(x) for x in nobs.tolist()]
        nobs = ["n: " + i for i in nobs]

        # Add it to the plot
        pos = range(len(nobs))
        for tick in pos:
            ax.text(ax.get_xlim()[1] * 1.01, pos[tick], nobs[tick],
                    horizontalalignment='left', size='medium', color='k')


def ret_on_stake(x):
    if x.stake.sum() < 0.1:
        return 0.
    else:
        return float(x.pnl.sum()) / x.stake.sum()


def total_pnl(x):
    return x.pnl.sum()


def ret_on_capital(x):
    if x.stake.sum() < 0.1 or len(x) == 0:
        print 0.
        return 0.
    else:
        max_capital = max(x.capital)
        print float(max_capital) * float(x.pnl.sum()) / (
                x.capital * x.stake.sum())
        return float(max_capital) * float(x.pnl.sum()) / (
                x.capital * x.stake.sum())