import warnings
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import LeavePGroupsOut

"""
References:
[1] Bailey, de Prado. "The Deflated Sharpe Ratio: Correcting for Selection
    Bias, Backtest Overfitting, and Non-Normality"
[2] Bailey, Borwein, de Pradoz, Zhu. "The Probability of Backtest Overfitting"
"""


# TODOs:
# - allow df_returns to have timestamp information to be able to add
#   time-based metrics such as annualized sharpe ratio, and will maybe
#   allow financial_metrics to be passed as a metric in _compute_metrics.
# - don't count nan returns which will mean "no position hold by the strategy"
# - add method in appendix A.3 of [1] to estimate the number of independent
#   trials implied by the total number of trials and their average correlation


def compute_psr(x, n_returns, moments):
    """Compute the Probabilistic Sharpe Ratio (PSR).

    PSR(x) is the probability that the true sharpe ratio is larger than x.
    It is estimated using the first four moments of the distribution of
    returns.

    Parameters
    ----------
    x : float,
        the reference sharpe ratio threshold

    n_returns : int,
        the length of the returns series

    moments : array-like, shape (4, )
        mean, standard deviation, skewness and kurtosis (Pearson's definition,
        ie normal ==> 3.0) of the distribution of returns.

    Returns
    -------
    psr : float in (0, 1)
        PSR(x)

    References
    ----------
    see [1]
    """

    if len(moments) != 4:
        raise ValueError('Expected the first 4 moments of the returns '
                         'distribution, got {}'.format(moments))

    sharpe_ratio = moments[0] / float(moments[1])

    psr = (sharpe_ratio-x) * np.sqrt(n_returns-1)
    psr /= np.sqrt(1 - sharpe_ratio*moments[2]
                   + sharpe_ratio**2 * (moments[3]-1)/4.)
    psr = stats.norm.cdf(psr)
    return psr


def compute_dsr(n_trials, n_returns, sharpes_std, moments):
    """Compute the Deflated Sharpe Ratio (DSR)

    DSR is the probability that the true sharpe ratio is larger than sr0,
    where sr0 is the expected (under the null hypothesis, ie null sharpe
    ratio) maximum sharpe ratio of n_trials strategies.

    sr0 is then just an adjusted rejection threshold taking into account
    the multiplicity of testing.

    Parameters
    ----------
    n_trials : int,
        the (honest) number of independent trials

    sharpes_std : float,
        standard deviation of the sharpe distributions

    n_returns : int,
        the length of the returns series

    moments : array-like, shape (4, )
        mean, standard deviation, skewness and kurtosis (Pearson's definition,
        ie normal ==> 3.0) of the distribution of returns.

    Returns
    -------
    dsr : float in (0, 1)

    References
    ----------
    see [1]
    """

    if n_trials < 10:
        warnings.warn("statistical approximation of the maximum sharpe ratio "
                      "needs larger sample size (n_trials > 10)")

    # compute sr0, the variance term of the expected maximum sharpe ratio:
    gamma = getattr(np, 'euler_gamma',
                    0.57721566490153286060)  # Euler-Mascheroni constant
    sr0 = ((1-gamma) * stats.norm.ppf(1 - 1. / n_trials)
           + gamma * stats.norm.ppf(1 - 1. / (n_trials*np.e)))
    sr0 *= sharpes_std

    dsr = compute_psr(sr0, n_returns, moments)
    return dsr


def dsr(df_returns, n_trials=None, n_trials_method=None, sharpes_std=None):
    """Compute the Deflated Sharpe Ratio (DSR)

    To reject the null hypothesis (sharpe ratio of 0) with 0.05 confidence we
    should ask for a DSR > 0.95

    Parameters
    ----------
    df_returns : np array or dataframe, shape (n_returns, n_trials)
        n_returns is the length of the returns time series and n_trials is
        the number of trials (number of tested strategies).

    n_trials : int, optional
        number of independent trials
        Specifying a lower number of trials than provided is thus possible
        - e.g to take into account correlation between strategies.
        Only one of n_trials and n_trials_method is allowed to be specified.
        If neither n_trials nor n_trials_method are provided, use the number of provided trials
        (n_columns of df_returns).

    n_trials_method: str, optional
        string that describes method to use to calculate the effective number of independent trials
        from df_returns (see get_effective_n_trials).
        Only one of n_trials and n_trials_method is allowed to be specified.
        If neither n_trials nor n_trials_method are provided, use the number of provided trials
        (n_columns of df_returns).

    sharpes_std : positive float, optional
        standard deviation of the sharpe distribution (across
        the trials).
        If not specified, use the provided trials to compute it.

    Returns
    -------
    dsrs : array of shape (n_trials,)
        deflated sharpe ratios for each provided trial.

    References
    ----------
    see [1]
    """

    if df_returns.ndim != 2:
        raise ValueError(
            "Expected 2D array, got {}D array instead."
            .format(df_returns.ndim))

    if n_trials is not None and not isinstance(n_trials, int):
        raise ValueError('n_trials should be an int if specified but it is {} ({})'.\
            format(type(n_trials), n_trials))

    if n_trials_method is not None and not isinstance(n_trials_method, str):
        raise ValueError('n_trials_method should be a str if specified but it is {} ({})'.\
            format(type(n_trials_method), n_trials_method))

    n_returns, n_trials_provided = df_returns.shape
    n_returns -= np.isnan(df_returns).sum(axis=0)

    if n_trials is None and n_trials_method is None:
        n_trials_ = n_trials_provided
        if sharpes_std is not None:
            raise ValueError("You should provide either both n_trials and "
                             "sharpes_std or none of them")
    elif n_trials is not None and n_trials_method is None:
        n_trials_ = n_trials
    elif n_trials is None and n_trials_method is not None:
        n_trials_, _ = get_effective_n_trials(df_returns, how=n_trials_method)
    elif n_trials is not None and n_trials_method is not None:
        raise ValueError('Only one of n_trials and n_trials_method should be used.')

    if n_trials_ < 2:
        raise ValueError(
            "n_trials={}, while a number larger than 2 is required "
            "Please specify n_trials parameter or increase the number "
            "of trials (columns) in your dataframe".format(n_trials_))

    moments = np.array([
        np.nanmean(df_returns, axis=0),
        np.nanstd(df_returns, axis=0, ddof=1),
        stats.skew(df_returns, axis=0, nan_policy='omit'),
        stats.kurtosis(df_returns, fisher=False, axis=0, nan_policy='omit')])

    sharpes = np.divide(moments[0], moments[1])

    if sharpes_std is None:
        sharpes_std = sharpes.std()
        if isinstance(n_trials, int):
            warnings.warn("You have provided a custom n_trials - do you also want to provide a "
                          "different sharpes_std?")

    dsrs = [compute_dsr(n_trials_, n_r, sharpes_std, moments[:, i])
            for i, n_r in enumerate(n_returns)]
    return np.array(dsrs)


def pbo(df_returns, n_splits=16, n_jobs=1, metric="sharpe", n_days=365):
    """Compute the probability of backtest overfitting (PBO)

    Parameters
    ----------
    df_returns : np array or dataframe, shape (n_returns, n_trials)
        n_returns is the length of the returns time series and n_trials is
        the number of trials (number of tested strategies).

    n_splits : int, optional (default=16)
        number of folds. Must be a positive even number.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel.
        If -1, then the number of jobs is set to the number of cores.

    metric : str or callable, optional
        name of the metric to be computed or function to compute it.
        If string, possible values are ["sharpe", "annualised_sharpe",
        "sortino", "annualised_sortino", "pnl"]

    n_days: int, optional (default=365)
        Number of days spanned by the returns. It is used in case
        metrics is annualised_sharpe or annualised_sortino to compute the
        average number of trades made by the strategy in a year.

    Returns
    -------
    proba_overfit : float in (0, 1)
        probability of overfitting (pbo)

    proba_loss : float in (0, 1)
        probability of loss (prob that metric -e.g sharpe- is < 0)

    lambda_logits: numpy array, shape(n_combinations,)
        lambda logits for each of the combinations. A negative lambda means that
        the best in-sample strategy performed worse than median out of sample.

    metrics_test: numpy array, shape(n_combinations,)
        metric of the best in-sample strategy computed out of sample per each
        combination

    metrics_train: numpy array, shape(n_combinations,)
        metrics of the best in-sample strategy per each combination

    References
    ----------
    see [2]
    """

    if df_returns.ndim != 2:
        raise ValueError(
            "Expected 2D array, got {}D array instead."
            .format(df_returns.ndim))

    _, n_trials = df_returns.shape

    if n_trials < 2:
        raise ValueError(
            "n_trials={}, while a number larger than 2 is required "
            "Please increase the number of trials (columns) in your "
            "returns dataframe")

    cscv = CSCV(n_splits=n_splits)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_compute_oos_scores)(
            df_returns, train_index, test_index, n_trials,
            metric, n_days)
        for train_index, test_index in cscv.split(df_returns))

    scores = np.array(scores)
    lambda_logits = scores[:, 0]  # could be named lambda_logits_test
    metrics_test = scores[:, 1]
    metrics_train = scores[:, 2]

    # prob of loss:
    proba_loss = np.array([1 if i <= 0 else 0 for i in metrics_test]).mean()
    # pbo:
    proba_overfit = np.array([1 if i <= 0 else 0
                              for i in lambda_logits]).mean()

    return proba_overfit, proba_loss, lambda_logits, metrics_test, metrics_train


def get_effective_n_trials(df_returns, how='mean_corr'):
    """Approximate the effective number of independent trials

        Parameters
        ----------
        df_returns : np array or dataframe, shape (n_returns, n_trials)
            n_returns is the length of the returns time series and n_trials is
            the number of trials (number of tested strategies).

        how : str, optional
            method to be used to estimate the effective number of trials
            At the moment, only 'mean_corr' is implemented, which uses the average correlation
            method described in [1] (A.3)

        Returns
        -------
        n_trials_eff : the effective number of trials
        info: other intermediate calculations we want to return (e.g. average correlation)


        References
        ----------
        see [1]
        """

    if how == 'mean_corr':
        corr_df = df_returns.fillna(0).corr()
        n_trials = len(corr_df)
        rho = (corr_df.sum().sum() - n_trials) / (n_trials * (n_trials - 1))
        n_trials_eff = rho + (1 - rho) * n_trials
        info = {'mean_corr': rho}
        return n_trials_eff, info
    else:
        raise ValueError('Method {} for calculating effective number of trials not recognised.'
                         .format(how))


def _compute_oos_scores(df_returns, train_index, test_index, n_trials,
                        metric="sharpe", n_days=365):
    """Compute OOS scores of the best IS strategy.

    Scores are :
    - lambda_logits as defined in [2], ie the logit of the normalized OOS rank
      of the best IS strategy. lambda_logit > 0 means that the best IS strategy
      is better than the (OOS) median strategy.

    - metric_test, eg the OOS score (e.g. sharpe) of the best IS strategy.


    Parameters
    ----------
    df_returns : np array or dataframe, shape (n_returns, n_trials)
        n_returns is the length of the returns time series and n_trials is
        the number of trials (number of tested strategies).

    train_index : array-like, shape (n_train_returns,)
        Indices of training samples.

    test_index : array-like, shape (n_test_returns,)
        Indices of test samples.

    n_trials : int, optional
        number of independent trials.

    metric : str or callable, optional
        name of the metric to be computed or function to compute it.
        If string, possible values are ["sharpe"]

    n_days: int, optional (default=365)
        Number of days spanned by the returns. It is used in case
        metrics is annualised_sharpe or annualised_sortino to compute the
        average number of trades made by the strategy in a year.

    Returns
    -------
    lambda_logit : float,
        OOS lambda_logit of the best IS strategy (bigger is better).

    metric_test : float,
        OOS score (e.g. sharpe) of the best IS strategy.
    """
    df_train = _indexing(df_returns, train_index)
    df_test = _indexing(df_returns, test_index)

    # in the pbo the train/test splitting is always 50/50
    train_ratio = 0.5
    test_ratio = 0.5
    metrics_train = compute_metrics(df_train, metric, int(train_ratio * n_days))
    metrics_test = compute_metrics(df_test, metric, int(test_ratio * n_days))

    best_trial_train = np.argmax(metrics_train)
    if np.isnan(best_trial_train):
        print 'WARNING: The best strategy has a Nan metric'
    # OOS metric of the best IS trial:
    metric_test = metrics_test[best_trial_train]

    # OOS rank of the best IS trial (2 argsort to get the rank):
    rank_test = metrics_test.argsort().argsort()[best_trial_train] + 1
    # adjusted rank_test
    rank_test = rank_test / (n_trials + 1.)
    # logit transform
    lambda_logit = np.log(rank_test / (1. - rank_test))

    return lambda_logit, metric_test, metrics_train[best_trial_train]


def compute_metrics(df_returns, metric="sharpe", n_days=365):
    """Compute the desired metric along axis=0

    Parameters
    ----------
    df_returns : np array or dataframe, shape (n_returns, n_trials)
        n_returns is the length of the returns time series and n_trials is
        the number of trials (number of tested strategies).

    metric : str or callable, optional
        name of the metric to be computed or function to compute it.
        If string, possible values are ["sharpe", "annualised_sharpe",
        "sortino", "annualised_sortino", "pnl"]

    n_days: int, optional (default=365)
        Number of days spanned by the returns. It is used in case
        metric is annualised_sharpe or annualised_sortino

    Returns
    -------
    metrics : array of shape (n_trials,)
        metric along x-axis.
    """

    non_null_values = len(df_returns) - np.isnan(df_returns).sum(axis=0)

    if metric in ['sharpe', 'annualised_sharpe']:
        computed_metrics = np.divide(np.nanmean(df_returns, axis=0),
                                     np.nanstd(df_returns, axis=0, ddof=1))

    elif metric in ['sortino', 'annualised_sortino']:

        # this needs to be done this way in case df_returns is a numpy array
        downside_std = np.nanstd(
            np.where(df_returns < 0, df_returns, np.nan), axis=0, ddof=1)

        computed_metrics = np.divide(np.nanmean(df_returns, axis=0),
                                     downside_std)

    elif metric == "pnl":
        computed_metrics = np.nansum(df_returns, axis=0)

    elif isinstance(metric, str):
        raise ValueError('metric {} unknown, feel free to add it here'
                         .format(metric))
    elif callable(metric):
        computed_metrics = np.apply_along_axis(metric, 0, df_returns)
    else:
        raise ValueError('Expected str or callable for metric argument, '
                         'got {}'.format(metric))

    if metric in ['annualised_sharpe', 'annualised_sortino']:
        annualised_factor = np.sqrt(365. * non_null_values / n_days)
        computed_metrics = np.multiply(annualised_factor, computed_metrics)

    computed_metrics[non_null_values < 3] = np.nan

    return np.array(computed_metrics)


class CSCV(LeavePGroupsOut):
    """Combinatorially Symmetric Cross-Validation used to estimate PBO.

    Parameters
    ----------
    n_splits : int, optional (default=16)
        number of folds. Must be a positive even number.

    References
    ----------
    see [2]
    """

    def __init__(self, n_splits=16):

        if n_splits % 2 or n_splits < 2:
            raise ValueError("The number of splits must be a positive "
                             "even number.")
        self.n_splits = n_splits
        self._group_labels = None
        self.n_returns = None

        super(CSCV, self).__init__(n_groups=n_splits//2)

    def split(self, X):
        """Generate indices to split data into training and test set.

        Notes
        -----
        The first ``n_returns % n_splits`` folds have size
        ``n_returns // n_splits + 1``, other folds have size
        ``n_returns // n_splits``, where ``n_returns`` is the number of
        returns.

        Parameters
        ----------
        X : np array or dataframe, shape (n_returns, n_trials)
            Training data, where n_returns is the length of the returns time
            series and n_trials is the number of trials (number of tested
            strategies).

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if X.ndim != 2:
            raise ValueError(
                "Expected 2D array, got {}D array instead."
                .format(X.ndim))

        self.n_returns, _ = X.shape
        groups = self._groups
        for train_index, test_index in super(CSCV, self).split(X, y=None,
                                                               groups=groups):

            yield train_index, test_index

    @property
    def _groups(self):
        """Return group labels for the returns used by LeavePGroupsOut while
        splitting the dataset into train/test set.
        """
        if self._group_labels is None:
            self._group_labels = self._create_groups()
        return self._group_labels

    def _create_groups(self):
        groups = -1 * np.ones(self.n_returns)

        n_splits = self.n_splits
        fold_sizes = (self.n_returns // n_splits) * np.ones(n_splits,
                                                            dtype=np.int)

        # to use all the returns, increase by 1 the size of first groups:
        fold_sizes[:self.n_returns % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            groups[start:] += 1
            current = stop
        return groups


def _indexing(X, indices):
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    elif hasattr(X, "take"):
        return X.take(indices, axis=0)
    else:
        raise ValueError("Expected a numpy array or pandas DataFrame, "
                         "got {}".format(type(X)))
