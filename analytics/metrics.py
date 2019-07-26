import datetime as dt
import numbers
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.index import DatetimeIndex
from scipy.stats import gennorm
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


def calc_trades(df):
    temp_out = {
        'principal_odds': df['matchedOdds'].values[0] if 'matchedOdds' in df.columns else df['odds'].values[0],
        'trade_won': df['pnl'].sum() > 0,
        'trade_return': df['pnl'].sum() == 0,
        'trade_lost': df['pnl'].sum() < 0,
        'return_on_stake': df['pnl'].sum() / df['stake'].sum(),
        'total_stake': df['stake'].sum(),
        'total_pnl': df['pnl'].sum()
    }
    return pd.Series(temp_out)


def calc_trades_stats(df, groups=None):
    def f(x):

        # average_odds = x.loc[x['return_on_stake'] > 0., 'principal_odds'].mean()
        average_odds = x.principal_odds.mean()
        target_success = 1. / average_odds

        n_won = x.trade_won.sum()
        n_lost = x.trade_lost.sum()

        won_ids = x.trade_won == True
        lost_ids = x.trade_lost == True

        av_win = x.loc[won_ids, 'total_pnl'].mean()
        av_loss = x.loc[lost_ids, 'total_pnl'].mean()

        # average_odds = 1. + av_win
        # target_success = 1./average_odds

        if n_lost > 0:
            cr = (n_won * av_win) / (- n_lost * av_loss)
        else:
            cr = np.nan

        p_success = np.nan
        if n_won + n_lost > 0:
            p_success = float(n_won) / (n_won + n_lost)

        out = {'n_predictions': len(x.index),
               'n_win': n_won,
               'n_push': x.trade_return.sum(),
               'n_lost': n_lost,
               'cr': cr,
               'prob_success': p_success,
               'mean_ros': x.total_pnl.sum() / x.total_stake.sum(), #x.return_on_stake.mean(),
               'total_stakes': x.total_stake.sum(),
               }
        return pd.Series(out)

    if groups is None:
        df_out = f(df.reset_index())
        df_out = df_out.to_frame().T
        return df_out
    return df.reset_index().groupby(groups).apply(f)


def create_summary_table(df_in, name):

    trade_df = df_in.groupby(['fixture_id']).apply(calc_trades)
    summary_table = calc_trades_stats(trade_df)
    summary_table = summary_table.reset_index()

    summary_table['strategy_short_name'] = name

    columns = [u'strategy_short_name',
               u'n_predictions', u'n_win', u'n_push', u'n_lost',
               u'prob_success', u'cr',
               u'mean_ros', u'total_stakes'
               ]

    summary_table = summary_table[columns]

    return summary_table.reset_index(drop=True)


def financial_metrics(_df, return_agg_by, groupby=None, capital=None, trade_agg_periods=None,
                      trading_periods=None, target_vol=None, odds_col='odds', dt_col='dt'):
    """
    Calculate financial metrics by looking at returns on capital
    Parameters
    ----------
    _df: a dataframe with columns: dt(datetime), pnl, stake, capital, odds, is_back
    return_agg_by: (str) how to aggregate the returns. by date or by fixture_week, etc...
    groupby: if not None are the columns of the df to groupby
    capital: what capital value to use if it is not specified in the df
    trade_agg_periods: (list of Numbers or date) The list of trading periods we should consider
     warning : this is used using pd.Series.reindex function so this could exclude days if not mentioned
    trading_periods: (float) number of trading periods in a year to calculate annualised_volatility
    target_vol: Annualised target volatility. If groupby not None, each group will target this volatility
        it does not affect the results but simply add a scale number to multiply your stakes with to target
        this volatility. See test_financial_metrics_month_to_date_aggr_by_date for an example
    odds_col: name of the columns containing the matched odds.
        - can be None if the liability is specified and you do not need to check that odds > 1.
    dt_col: name of the columns containing the datetime information

    Returns (DataFrame) financial metrics, one row per group (one row if no groups)
    -------

    """

    if _df.empty:
        return None

    df, constant_capital = _format_df_for_capital_metrics(_df, capital, odds_col, dt_col)

    def _gr_f(x, constant_capital, return_agg_by, target_vol=None, trade_agg_periods=None,
              trading_periods=None):
        if return_agg_by not in x:
            raise Exception('column {} for aggregating returns is not in the dataframe'.format(return_agg_by))
        out = OrderedDict()
        out['n_trades'] = len(x.index)

        if 'event_id' in x or 'fixture_id' in x:
            event_id_ = 'event_id' if 'event_id' in x else 'fixture_id'
            epnl = x.groupby(event_id_)['return'].sum()
            out['n_events'] = len(x[event_id_].unique())
            out['cr_event'] = -epnl[epnl > 0].sum() / epnl[epnl < 0].sum()
            out['event_sharpe_ratio'] = np.sum(epnl) / (np.std(epnl) * np.sqrt(out['n_events']))
            out['event_sortino'] = np.sum(epnl) / (np.std(epnl[epnl < 0]) * np.sqrt(out['n_events']))


        out['n_win'] = (x['return'] > 0).sum().astype(int)
        out['n_loss'] = (x['return'] < 0).sum().astype(int)
        out['hit_ratio'] = out['n_win'].astype(float) / (out['n_win'] + out['n_loss'])

        out['average_trade_win'] = x.loc[x['return'] > 0, 'return'].mean()
        out['average_trade_loss'] = - x.loc[x['return'] < 0, 'return'].mean()

        #out['unitary_stake_return'] = x['return'].mean() / capital_fraction
        out['unitary_stake_return'] = (x['pnl']/ x['liability']).mean()

        out['cr_trade'] = (float(out['n_win']) * out['average_trade_win']) / (
        float(out['n_loss']) * out['average_trade_loss'])

        return_agg = x.groupby(return_agg_by)['return'].sum()
        return_agg = pd.DataFrame(return_agg)
        if len(return_agg.index) == 0:
            return pd.Series({})

        if trade_agg_periods is not None and len(trade_agg_periods) > 0:
            if len(set(return_agg.index.values).intersection(set(trade_agg_periods))) == 0:
                raise Exception('The specified trade_agg_periods do not match any of the existing periods.')

            if isinstance(trade_agg_periods[0], dt.date) or isinstance(trade_agg_periods[0], dt.datetime):
                new_index = DatetimeIndex(trade_agg_periods)
            else:
                new_index = pd.Index(trade_agg_periods)

            return_agg = return_agg.reindex(new_index, fill_value=0)

        # Note: this implementation of cum_return is this way after much much discussion between DE and MW
        # Essentially we decided compounding doesn't make much sense when doing active capital management
        # And that daily returns won't be the official unit we report to investors (likely will be monthly returns)
        return_agg['cum_return'] = return_agg['return'].cumsum()
        d_idx = return_agg['return'] < 0

        av_agg_win = return_agg.loc[return_agg['return'] > 0, 'return'].mean()
        av_agg_loss = - return_agg.loc[return_agg['return'] < 0, 'return'].mean()
        n_win_agg = (return_agg['return'] > 0).sum()
        n_lose_agg = (return_agg['return'] < 0).sum()
        n_periods = len(return_agg.index)

        out['cr_day'] = (av_agg_win * float(n_win_agg)) / (av_agg_loss * float(n_lose_agg))

        # TODO There's an additional thing to consider when we look at the portfolio,
        # TODO we have to aggregate across all strategies and see where trade days overlap,
        # TODO and when they don't, so you could have 2 strategies in a portfolio, each with 100 trade days per year
        # TODO and the portfolio could have anywhere between 100 and 200 trade days per year...
        cum_return = (return_agg.tail(1)['cum_return'].values[0])
        volatility = (return_agg['return'].std()) * np.sqrt(n_periods)
        downside_volatility = return_agg.loc[d_idx, 'return'].std() * np.sqrt(n_periods)

        if trading_periods is not None:
            # TODO consider using .std() instead of np.std to be consistent with rest of the function
            # pandas uses unbiased estimator whereas numpy does not
            # https://stackoverflow.com/questions/24984178/different-std-in-pandas-vs-numpy
            out['volatility_annualised'] = np.std(return_agg['return']) * np.sqrt(trading_periods)

        out['cum_return'] = cum_return
        out['volatility (not annualised)'] = volatility
        out['downside_volatility (not annualised)'] = downside_volatility

        if target_vol is not None:
            # we have the volatility, we can readjust to reach the target
            if return_agg_by == 'date':
                if 'volatility_annualised' in out:
                    scale = target_vol / out['volatility_annualised']
                else:
                    scale = target_vol * np.sqrt((_df.dt.max() - _df.dt.min()).days / 365.25) / out['volatility (not annualised)']
            else:
                raise NotImplementedError('Could not target vol if ret_agg_by is not date')
            out['scale'] = scale

        if out['volatility (not annualised)'] == 0.:
            out['sharpe_ratio'] = np.nan
        else:
            out['sharpe_ratio'] = out['cum_return'] / out['volatility (not annualised)']

        if out['downside_volatility (not annualised)'] == 0.:
            out['sortino_ratio'] = np.nan
        else:
            out['sortino_ratio'] = out['cum_return'] / out['downside_volatility (not annualised)']

        period = 'days' if return_agg_by == 'date' else return_agg_by
        out['maximum_drawdown'], out['drawdown_duration ({})'.format(period)] = max_dd(return_agg['cum_return'] + 1.)
        out['maximum_runup'], out['runup_duration ({})'.format(period)] = max_runup(return_agg['cum_return'] + 1.)

        if constant_capital:
            out['total_pnl'] = out['cum_return'] * df.iloc[0]['capital']
        else:
            out['total_pnl'] = x['pnl'].sum()

        out['n_trading_days'] = n_periods

        return pd.Series(out)


    args_gr_f = {'constant_capital': constant_capital,
                 'return_agg_by': return_agg_by,
                 'target_vol': target_vol,
                 'trade_agg_periods': trade_agg_periods,
                 'trading_periods': trading_periods}
    if groupby is None:
        df_out = _gr_f(df, **args_gr_f)
        df_out = df_out.to_frame().T
    else:
        df_out = df.groupby(groupby).apply(_gr_f, **args_gr_f)

    if len(df_out.index) == 0:
        return None

    for c in ['n_trades', 'n_win', 'n_loss']:
        df_out[c] = df_out[c].apply(lambda xx: int(xx) if not np.isnan(xx) else np.nan)

    return df_out


def flat_capital_metrics(_df, groupby=None, capital=None, trade_dates=None, trading_days=None,
                         odds_col='odds', dt_col='dt'):
    return financial_metrics(_df, 'date', groupby=groupby, capital=capital, trade_agg_periods=trade_dates,
                             trading_periods=trading_days, target_vol=None, odds_col=odds_col,
                             dt_col=dt_col)

def _format_df_for_capital_metrics(_df, capital, odds_col='odds', dt_col='dt'):
    if odds_col is not None:
        wrong_odds = (_df[odds_col] < 1.).sum()
        if wrong_odds > 0:
            raise Exception('Your dataframe has some odds < 1.')
    if 'liability' not in _df.columns:
        if odds_col is None:
            raise Exception('liability column not found and no odds_col '
                            'specified to compute it (odds_col=None)')
        _df['liability'] = _df.apply(get_liability, stake_col='stake',
                                                   odds_col=odds_col, axis=1)
    df = _df.sort_values(dt_col).copy()
    df['date'] = df[dt_col].apply(lambda x: x.date())

    constant_capital = False
    if 'capital' not in df.columns:
        if capital is None:
            raise Exception('No capital in dataframe or provided')
        else:
            if isinstance(capital, numbers.Number):
                df['capital'] = capital
                constant_capital = True
            else:
                df = pd.merge(df, capital, how='left', on=['date'])
    df['return'] = df['pnl'] / df['capital']
    return df, constant_capital


def outlier_threshold(x):
    q = np.percentile(x, [25, 75, 95])
    th1 = 2.5 * q[1] - q[0]
    th2 = q[2]
    return max(th1, th2)


def flat_capital_traces(_df, groupby=None, capital=None, return_agg_by='date', dt_col='dt'):
    """
    to calculate simple finance metrics given
    :param _df: a dataframe with columns: dt(datetime), pnl, stake
    :groupby: if not None are the columns of the df to groupby
    :capital_fraction: obvious
    :dt: name of the datetime column
    :return: dataframe with the compund_return and date
    """

    df = _df.sort(dt_col).copy()
    df['date'] = df[dt_col].map(lambda x: x.date())

    df_has_capital = 'capital' in df.columns
    if not df_has_capital:
        if capital is None:
            raise Exception('No capital in dataframe or provided')
        else:
            df['capital'] = capital

    df['return'] = df['pnl'] / df['capital']

    def _gr_f(x, return_agg_by):

        daily_agg = x.groupby(return_agg_by)['return'].sum().reset_index().set_index(return_agg_by)
        daily_agg['cum_return'] = daily_agg[['return']].cumsum()
        if return_agg_by == 'date':
            daily_agg.index = pd.to_datetime(daily_agg.index)
        daily_agg.index = daily_agg.index.rename(return_agg_by)

        return daily_agg[['cum_return']]

    if groupby is None:
        return _gr_f(df, return_agg_by=return_agg_by)
    else:
        return df.groupby(groupby).apply(_gr_f, return_agg_by=return_agg_by)


def cumulative_capital_traces(_df, starting_capital, groupby=None):
    """
    Calculate the capital trace from already compounded trades
    :param _df: a dataframe with columns: dt(datetime), pnl, stake
    :param starting_capital: if not None are the columns of the df to groupby
    :param groupby" if not None are the columns of the df to groupby
    :return: dataframe with the compund_return and date
    """

    df = _df.sort('dt').copy()
    df['date'] = df['dt'].map(lambda x: x.date())

    def _gr_f(x):
        daily_agg = x.groupby('date')[['return']].sum().sort_index()
        daily_agg['cum_return'] = daily_agg['return'].cumsum() / starting_capital
        return daily_agg[['cum_return']]

    if groupby is None:
        return _gr_f(df)
    else:
        return df.groupby(groupby).apply(_gr_f)


def max_dd(ser):
    i = np.argmax(np.maximum.accumulate(ser) - ser)  # end of the period
    j = np.argmax(ser[:i])  # start of period

    dd_length = (i - j)
    try:
        dd_length = int(dd_length.days)
    except AttributeError:
        pass
    # we no longer do a % change just the absolute difference of the compound returns (obviously the +1s cancel out)
    dd_fraction = (ser[j] - ser[i])
    # print "date=%s peak=%.3f" % (j, ser[j])
    # print "date=%s low=%.3f " % (i, ser[i])

    return -dd_fraction, dd_length


def max_runup(ser):
    diff = ser - np.minimum.accumulate(ser)
    if (diff == 0).all():
        # case where it keeps losing, no run up
        return 0, 0

    i_r = np.argmax(diff)

    j_r = np.argmin(ser[:i_r])

    ru_fraction = (ser[i_r] - ser[j_r]) / ser[j_r]
    ru_length = i_r - j_r
    try:
        ru_length = int(ru_length.days)
    except AttributeError:
        pass

    return ru_fraction, ru_length


def get_backtest_results_ready(_df):
    df = _df.copy()
    df['date'] = df['dt'].map(lambda x: x.date())
    if 'liability' not in _df.columns:
        df['liability'] = _df.apply(get_liability, axis=1)
    return df


def get_liability(x, stake_col='stake', odds_col='odds'):
    return x[stake_col] if x['is_back'] else x[stake_col]*(x[odds_col] - 1.)


def get_equivalent_odds(x):
    return x['odds'] if x['is_back'] else x['odds'] / (x['odds'] - 1.)


def get_equivalent_back_df(df):
    """
    Convert any lay trades in df by its equivalent back trade

    Warnings
    --------
    It only updates the odds, stake and liability !!
    It does not change the selection or other columns, the pnl will tell us
        whether it was losing or winning regardless of selection


    Parameters
    ----------
    df: DataFrame with columns "odds" and "is_back"
    """
    df['liability'] = df.apply(get_liability, axis=1)
    df['odds'] = df.apply(get_equivalent_odds, axis=1)
    df['is_back'] = True
    df['stake'] = df['liability']
    return df


def default_aggr_trades(trades, aggr_cols):
    mean_cols = []
    sum_cols = ['stake', 'pnl']
    first_cols = []
    unique_cols = ['dt']
    stake_wm_cols = ['odds']
    stake_col = 'stake'
    return aggr_trades(trades, aggr_cols, mean_cols=mean_cols,
                       sum_cols=sum_cols, first_cols=first_cols,
                       unique_cols=unique_cols,
                       stake_weighted_mean_cols=stake_wm_cols,
                       stake_col=stake_col)


def aggr_trades(trades, aggr_cols, mean_cols, sum_cols, first_cols,
                unique_cols, stake_weighted_mean_cols, stake_col='stake'):
    """
    Warnings
    --------
    You need to handle yourself beforehand if you have back and lay trades
    The function will raise an error if you don't handle it correctly

    Parameters
    ----------
    trades: (DataFrame) trades
    aggr_cols: (list of str) however you want to aggregate trades.
        usually thing is per sticker for example
    mean_cols: (list of str) will use the method mean of trades for each
        column in here (apply to each aggr_by group)
    sum_cols: (list of str) will use the method sum of trades for each
        column in here (apply to each aggr_by group)
    first_cols: (list of str) will use the method first of trades for each
        column in here (apply to each aggr_by group)
    unique_cols: (list of str) will make sure that all the values within each
        aggr_by group is unique and will apply first()
    stake_weighted_mean_cols: (list of str) will provide a stake weighted mean
        using the stake_col column
    stake_col: (str) column of the stake used for the stake weighted mean

    Returns
    -------
    (DataFrame) Aggregated trades

    """
    if 'is_back' not in trades:
        raise ValueError('boolean is_back column should be in trades df')

    if 'is_back' not in (aggr_cols + unique_cols):
        # we are not grouping by back/lay, and is not in unique.
        if 'is_back' in (mean_cols + sum_cols + first_cols +
                             stake_weighted_mean_cols):
            warnings.warn('WARNING !! back and lay trades together in the '
                          'DataFrame is unexpected. Make sure you know what'
                          ' you are doing !')
        else:
            # add 'is_back' to the unique cols to check we have indeed
            # only back or lay
            unique_cols.append('is_back')

    aggr_fun = lambda trades: _fun_aggr_back_trades(
        trades, mean_cols=mean_cols, sum_cols=sum_cols, first_cols=first_cols,
        unique_cols=unique_cols,
        stake_weighted_mean_cols=stake_weighted_mean_cols, stake_col=stake_col)
    return trades.groupby(aggr_cols).apply(aggr_fun).reset_index()


def _fun_aggr_back_trades(trades, mean_cols, sum_cols, first_cols, unique_cols,
                          stake_weighted_mean_cols, stake_col='stake'):
    for col in unique_cols:
        if len(trades[col].unique()) > 1:
            raise ValueError('Value for column {} not unique : {}'.format(
                col, trades[col].unique().tolist()))
    out = {}
    for col in mean_cols:
        out[col] = trades[col].mean()
    for col in sum_cols:
        out[col] = trades[col].sum()
    for col in first_cols + unique_cols:
        out[col] = trades[col].iloc[0]
    for col in stake_weighted_mean_cols:
        out[col] = ((trades[col] * trades[stake_col]).sum() /
                    trades[stake_col].sum())
    return pd.Series(out)


def aggregate_trades(df, cols_groupby):
    """
    This functions aggregates the trades, by first converting to an equivalent_back_df.
    :param df: contains: dt, stake, odds, pnl, capital, is_back, any_groupby_column
    :param cols_groupby: list of columns to groupby. Note that also groupsby 'win/loss' automatically
    :return: it will return a dataframe with [dt, stake, odds, pnl, capital, is_back, any_groupby_column]
    NB: it will delete any other column!
    """
    warnings.warn('Depreciated. Please use aggr_trades instead',
                  DeprecationWarning)

    df = get_equivalent_back_df(df)
    df['won'] = df['pnl'] > 0.
    cols_groupby = cols_groupby + ['won']
    cols = ['dt', 'is_back', 'stake', 'odds', 'pnl', 'capital'] + cols_groupby
    df = df[cols]

    def agg_fun(x):
        out = {
            'pnl': x.pnl.sum(),
            'stake': x.stake.sum(),
            'odds': (x.odds * x.stake).sum() / x.stake.sum(),
            'dt': x.dt.values[0],
            'is_back': True,
            'capital': x.capital.values[0],
            'n_bets': len(x.index)
               }
        return pd.Series(out)

    df = df.groupby(cols_groupby).apply(agg_fun).reset_index()

    return df


def make_date_list():
    start_date = '2015-07-29'
    stop_date = time.strftime('%Y-%m-%d')

    current_date = start_date

    date_list = []
    while current_date < stop_date:
        date_temp = {'start': current_date}
        current_date = dt.datetime.strptime(current_date, '%Y-%m-%d') + relativedelta(days=7)
        current_date = dt.datetime.strftime(current_date, '%Y-%m-%d')
        date_temp['stop'] = current_date
        date_list.append(date_temp)

    return date_list


def estimate_drawdown_for_probability(daily_returns, prob, trading_days=250, n=1000):
    """
    Calculates the drawdown corresponding to the probability passed in by fitting a general normal distribution
    and simulating returns for a year to generate a CDF (a fn of: independent var - drawdown , dependent var -
    P(drawdown >= prob)
    Note if you want to find the moments of the fitted distribution do:
    (mean, var, skew, kurtosis) = scipy.stats.gennorm.stats(b_sample, moments='mvsk')
    May in the future want to check if the distribution is a good fit or not
    :param daily_returns: time series of empirical daily returns
    :param prob: probability (float) for which to find the corresponding drawdown
    :param trading_days: the number of trading days expected in a year
    :param n: number of simulations to run
    :return: a drawdown number in percentage terms; will be negative (e.g. -0.21) and the parameter corresponding to
    the fitted generalized normal distribution
    """
    drawdown_indexes = [-float(i)/100. for i in range(0, 100)]
    if prob < 0 or prob > 1:
        raise ValueError('Parameter prob must be between 0 and 1')
    b, loc, scale = gennorm.fit(daily_returns)
    (mean, var, skew, kurtosis) = gennorm.stats(b, scale=scale, moments='mvsk')

    simulated_drawdowns = []
    datetime_index = [dt.datetime(2010, 1, 1) + dt.timedelta(days=t) for t in range(0, trading_days)]
    for i in range(0, n):
        simulated_daily_returns = gennorm.rvs(b, loc=loc, scale=scale, size=trading_days)
        s = pd.Series(simulated_daily_returns, index=datetime_index)
        return_series = s.cumsum()
        (max_drawdown, dd_length) = max_dd(return_series)
        simulated_drawdowns.append(max_drawdown)
    simulated_drawdowns_series = pd.Series(simulated_drawdowns)
    empirical_probabilities = []
    for d in drawdown_indexes:
        incidence_count = (simulated_drawdowns_series <= d).sum()
        empirical_probabilities.append(float(incidence_count) / float(n))

    #will have least negative drawdown first in drawdown_indexes and highest probability empirical_probabilities first
    #we are conservative: it will select the higher probability drawdown upon non-match (less negative / closer in)
    cdf_series = pd.Series(empirical_probabilities, index=drawdown_indexes)
    matching_drawdown = cdf_series[cdf_series.values >= prob].index[-1]
    return (matching_drawdown, cdf_series, b, loc, scale)


def adjust_sharpe_ratio(all_met_df, split_by=None, split_col_in_index=False):
    """
    Provides a new value of the sharpe ratio adjusted according to the number of results being looked at
    :param all_met_df: Dataframe with at least one column sharpe_ratio
    :param split_by: (list) columns for adjusting different groups differently
    :param split_col_in_index: (bool) whether at least one column in split_by is in the index
    warning : All the index levels must be named for this to work
    :return: the dataframe with an extra column corrected_sharpe_ratio

    Original idea from the paper [2014] Evaluating Trading Strategies, from CAMPBELL R. HARVEY AND YAN LIU
    https://faculty.fuqua.duke.edu/~charvey/Research/Published_Papers/P116_Evaluating_trading_strategies.pdf
    """
    if split_col_in_index:
        index_names = all_met_df.index.names
        all_met_df = all_met_df.reset_index()

    if split_by is not None and any([c not in all_met_df.columns for c in split_by]):
        raise Exception('At least one column in split_by is not found in columns '
                        '(split_by={})'.format(split_by))

    def _add_corrected_sharpe(mkt_df):
        if len(mkt_df) == 1:
            # it is independent test as it is alone, nothing to correct/adjust
            mkt_df.loc[:, 'corrected_sharpe_ratio'] = mkt_df['sharpe_ratio']
            return mkt_df

        # take the p_val of the sharpe ratios from the two-sided t student test
        mkt_df.loc[:, 'p_val'] = t.sf(np.abs(mkt_df['sharpe_ratio']), len(mkt_df) - 1) * 2
        mkt_df.loc[:, 'corrected_p_val'] = multipletests(mkt_df['p_val'])[1]
        mkt_df.loc[:, 'corrected_sharpe_ratio'] = t.isf(mkt_df['corrected_p_val'] / 2, len(mkt_df) - 1)
        mkt_df.loc[mkt_df.sharpe_ratio < 0, 'corrected_sharpe_ratio'] *= -1
        del mkt_df['p_val']
        del mkt_df['corrected_p_val']
        return mkt_df

    if split_by is None:
        all_met_df = _add_corrected_sharpe(all_met_df)
    else:
        all_met_df = all_met_df.groupby(split_by, as_index=False).apply(_add_corrected_sharpe)

    if split_col_in_index:
        all_met_df.set_index(index_names)
    return all_met_df

def calculate_metrics(strategy_df, n_days=None, underlying_price=None, capital=None):
    """Function to compute financial metrics for Crypto strategies.

    strategy_df is assumed to be a DataFrame containing all the trades made by the strategy. It is
    assumed that is has the following columns: entry_time, exit_time, entry_price, exit_price,
    is_long, pnl. The pnl here is assumed to take into account the size of the trade, so it does not
    have to be equal to exit_price - entry_price.

    Parameters:
    ----------
    strategy_df: dataframe
        dataframe containing the trades of the strategy. See above for more information about its
        structure

    n_days: int, optional (default=None)
        number of days spanned by the strategy_df dataframe. It is used to annualise sharpe and
        sortino ratios and volatility, by computing the average number of trades the strategy places
        in a year. If it is None, then the difference between the exit times of the last and first
        trades is used.

    underlying price: series, optional (default=None)
        price series of the underlying instrument. It is assumed to be indexed by time and that a
        merge can be performed with the exit time of the trades in strategy_df, so it has to be in
        the same frequency

    capital: float, optional (default=None)
        capital of the strategy. If it is not None all the pnls generated by the strategy are
        normalized by the capital, otherwise they are measured in dollar terms.

    Returns
    -------
        metric, dataframe containing a series of financial metrics for the considered strategy
    """

    if n_days is None:
        n_days = (strategy_df.exit_time.iloc[-1] - strategy_df.exit_time.iloc[0]).days

    out = OrderedDict()
    n_returns = len(strategy_df)
    # factor for annualising sharpe and sortino ratios
    annualise_factor = 365. * n_returns / n_days
    # getting the pnl series
    return_series = strategy_df.pnl

    # possible normalisation
    if capital is not None:
        return_series /= capital
        if min(return_series) < -1:
            msg = """WARNING: at least one of the returns of the strategy normalised by the capital "
                     is less than -1!"""
            print msg

    out['n_returns'] = n_returns
    out['n_positive_returns'] = (return_series > 0).sum().astype(int)
    out['n_negative_returns'] = (return_series < 0).sum().astype(int)
    out['hit_ratio'] = float(out['n_positive_returns']) / out['n_returns']
    out['av_returns_year'] = annualise_factor

    out['average_positive_return'] = return_series[return_series > 0].mean()
    out['average_negative_return'] = abs(return_series[return_series < 0].mean())
    out['average_return'] = return_series.mean()
    out['cr_trade'] = (float(out['n_positive_returns']) * out['average_positive_return']) / (
        float(out['n_negative_returns']) * out['average_negative_return'])

    out['cumulative_return'] = return_series.sum()
    out['volatility'] = return_series.std() * np.sqrt(annualise_factor)
    d_idx = (return_series < 0)
    out['downside_volatility'] = return_series.loc[d_idx].std() * np.sqrt(annualise_factor)
    out['mean_return'] = return_series.mean()
    out['std_return'] = return_series.std()
    out['downside_std_return'] = return_series[return_series < 0].std()

    out['sharpe_ratio'] = out['mean_return'] / out['std_return']
    out['annualised_sharpe_ratio'] = out['mean_return'] / out['std_return'] * np.sqrt(annualise_factor)
    out['sortino_ratio'] = out['mean_return'] / out['downside_std_return']
    out['annualised_sortino_ratio'] = out['mean_return'] / out['downside_std_return'] * np.sqrt(annualise_factor)

    out['maximum_drawdown'], out['drawdown_duration'] = max_dd(np.nan_to_num(return_series.values))
    out['maximum_runup'], out['runup_duration']= max_runup(np.nan_to_num(return_series.values))

    out['ratio_longs'] = float(len(strategy_df[strategy_df.is_long])) / n_returns
    holding_period = strategy_df.exit_time.subtract(strategy_df.entry_time)
    holding_period_hours = holding_period.apply(lambda x: x.total_seconds() / 3600)
    out['average_holding_period (hours)'] = holding_period_hours.mean()
    out['return_long'] = return_series[strategy_df.is_long].sum()
    out['return_short'] = return_series[~strategy_df.is_long].sum()

    if underlying_price is not None:
        underlying_returns = underlying_price.diff()
        underlying_returns.columns = ['price']
        returns_p = underlying_returns.diff()
        comparison = pd.merge(returns_p, strategy_df[['pnl', 'exit_time']], how='left',
                              left_index=True, right_on='exit_time').fillna(0)
        correlation = comparison.corr()
        out['correlation_underlying'] = correlation.loc['price', 'pnl']

    return pd.Series(out).to_frame()
