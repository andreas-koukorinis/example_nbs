import datetime as dt
import warnings
from collections import Counter, defaultdict, OrderedDict

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from bokeh.models import Tabs, Panel, Range1d, LinearAxis, ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models.widgets.tables import NumberFormatter
from bokeh.plotting import figure, show
from bokeh.layouts import column
from scipy.stats import gennorm, genlogistic, norm, johnsonsu, johnsonsb, gumbel_l, gumbel_r
from scipy.stats import skew, kurtosis
from sgmtradingcore.analytics.metrics import max_dd
from sklearn.cross_validation import KFold
from sklearn.neighbors import KernelDensity
from statsmodels.tsa.stattools import adfuller, acf

PARAMETRIC_DISTS = [gennorm, genlogistic, norm, johnsonsu, johnsonsb, gumbel_l, gumbel_r, norm]


# Class to provide a KDE fitting distribution
class KDEDist(object):
    def __init__(self, bw, kernel='gaussian'):
        self._bw = bw
        self._kernel = kernel
        self._kd = KernelDensity(bandwidth=bw, kernel=kernel)
        self._samples = None

    @staticmethod
    def bw_range(x, n=3):
        max_pwr = 2
        h_opt = np.std(x) * (4. / (3. * len(x))) ** 0.2
        pwrs = np.concatenate([np.linspace(-max_pwr, 0, n + 1), np.linspace(0, max_pwr, n + 1)[1:]])
        return h_opt * 2 ** pwrs

    @property
    def name(self):
        return 'KDE({}, {:.5f})'.format(self._kernel, self._bw)

    @property
    def samples(self):
        if self._samples is None:
            self._samples = self.rvs(100000)
        return self._samples

    def dist(self):
        return self

    def fit(self, x):
        self._kd.fit(np.reshape(x, (len(x), 1)))
        return self

    def logpdf(self, x):
        return self._kd.score_samples(np.reshape(x, (len(x), 1)))

    def rvs(self, n):
        return self._kd.sample(n).reshape(n)

    def stats(self, moments='mv'):
        out = []
        if 'm' in moments:
            out.append(np.array([np.mean(self.samples)]))
        if 'v' in moments:
            out.append(np.array([np.var(self.samples)]))
        if 's' in moments:
            out.append(np.array([skew(self.samples)]))
        if 'k' in moments:
            out.append(np.array([kurtosis(self.samples)]))
        return tuple(out)

    def ppf(self, q):
        return np.percentile(self.samples, q)


class DailyReturnVisualisations(object):
    def __init__(self, strategy_name, daily_returns, n_trading_days=250, n_drawdown_simulations=1000, n_folds=10):
        self.strategy_name = strategy_name
        self.daily_returns = np.array(daily_returns)
        self.n_trading_days = n_trading_days
        self.n_drawdown_simulations = n_drawdown_simulations

        self.choose_message = ''
        self.chosen_rv = None
        self.fitted_stats = {}

        # Choose the distribution
        self.choose_distribution(n_folds=n_folds)

        # Simulate the drawdowns
        self.drawdowns = []
        self._simulate_drawdowns()

        # Clear the status updates
        print '\r ',
        print

    def choose_distribution(self, n_folds=10):
        '''
        Function to fit several distributions and choose the best one by cross-validated maximum likelihood
        :return:
        '''
        if n_folds > len(self.daily_returns):
            raise ValueError('Number of folds must be at least the number of points')

        rv_classes = PARAMETRIC_DISTS + [KDEDist(bw) for bw in KDEDist.bw_range(self.daily_returns)]
        kf = KFold(len(self.daily_returns), n_folds=n_folds)
        scores = {'train': defaultdict(list), 'test': defaultdict(list)}
        fitted = {}
        stats = {}
        for rvc in rv_classes:
            print '\rEvaluating {}'.format(rvc.name),

            for train_idx, test_idx in kf:
                r_train = self.daily_returns[train_idx]
                r_test = self.daily_returns[test_idx]

                if isinstance(rvc, KDEDist):
                    rvc.fit(r_train)
                    rvi = rvc
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        params = rvc.fit(r_train)
                    rvi = rvc(*params)
                scores['train'][rvc.name].append(rvi.logpdf(r_train).sum())
                scores['test'][rvc.name].append(rvi.logpdf(r_test).sum())

            # Get stats after fitting on all data
            if isinstance(rvc, KDEDist):
                rvc.fit(self.daily_returns)
                rvi = rvc
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params = rvc.fit(self.daily_returns)
                rvi = rvc(*params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats[rvc.name] = rvi.stats(moments='mvsk')
                fitted[rvc.name] = rvi

        test_mean_scores = [(nn, np.mean(s)) for nn, s in scores['test'].iteritems()]
        test_mean_scores = sorted(test_mean_scores, cmp=lambda a, b: cmp(b[1], a[1]))

        self.fitted_stats = OrderedDict([(x[0], stats[x[0]]) for x in test_mean_scores])

        self.choose_message = '{} days, best fitting distribution is {} (cv llh = {:.1f})'
        self.choose_message = self.choose_message.format(len(self.daily_returns), *test_mean_scores[0])
        self.chosen_rv = fitted[test_mean_scores[0][0]]

    def _simulate_drawdowns(self):
        # Simulate drawdowns
        print '\r Simulating drawdowns',
        c = Counter([])
        for ii in range(self.n_drawdown_simulations):
            daily_ret = self.chosen_rv.rvs(self.n_trading_days)
            c += Counter(np.sign(daily_ret[1:] - daily_ret[:-1]))
            self.drawdowns.append(max_dd(pd.Series(
                np.cumsum(daily_ret),
                index=[dt.datetime(2010, 1, 1) + dt.timedelta(ii) for ii in range(self.n_trading_days)]))[0])

    def generate_summary_table(self):
        # Output tables of stats
        rows = ['Input data'] + self.fitted_stats.keys()
        means = [np.mean(self.daily_returns)] + [float(s[0]) for s in self.fitted_stats.values()]
        variances = [np.var(self.daily_returns)] + [float(s[1]) for s in self.fitted_stats.values()]
        skews = [skew(self.daily_returns)] + [float(s[2]) for s in self.fitted_stats.values()]
        kurtosises = [kurtosis(self.daily_returns)] + [float(s[3]) for s in self.fitted_stats.values()]
        raw_data = dict(rows=rows, means=means, variances=variances, skews=skews, kurtosises=kurtosises)

        source = ColumnDataSource(raw_data)
        number_formatter = NumberFormatter(format='-0.0000')
        columns = [
            TableColumn(field="rows", title="", width=200),
            TableColumn(field="means", title="Mean", width=80, formatter=number_formatter),
            TableColumn(field="variances", title="Variance", width=80, formatter=number_formatter),
            TableColumn(field="skews", title="Skewness", width=80, formatter=number_formatter),
            TableColumn(field="kurtosises", title="Kurtosis", width=80, formatter=number_formatter)
        ]
        data_table = DataTable(source=source, columns=columns, width=550, height=150, row_headers=False)
        return data_table

    def generate_stationarity_table(self):
        # Test for non-stationarity
        adf_res = adfuller(self.daily_returns)

        # Output tables of stats
        raw_data = dict(
            test_stat=[adf_res[0]],
            p_value=[adf_res[1]],
            conclusion=['non-stationarity = {} @ 5%'.format('REJECTED' if adf_res[1] < 0.05 else 'NOT REJECTED')]
        )
        source = ColumnDataSource(raw_data)
        number_formatter = NumberFormatter(format='-0.0000')
        columns = [
            TableColumn(field="test_stat", title="ADF Test Statistic", width=150),
            TableColumn(field="p_value", title="p-value", width=80, formatter=number_formatter),
            TableColumn(field="conclusion", title="Conclusion", width=400),
        ]
        data_table = DataTable(source=source, columns=columns, width=650, height=100, row_headers=False)
        return data_table

    def generate_simulated_drawdown_table(self):
        pp = [50, 30, 20, 10, 5, 1]
        ddp = np.percentile(self.drawdowns, pp)

        pp_formatted = [float(p) / 100. for p in pp]

        source = ColumnDataSource(dict(
            probabilities=pp_formatted,
            drawdowns=ddp
        ))
        probability_formatter = NumberFormatter(format='0.00')
        drawdown_formatter = NumberFormatter(format='(0.00)%')
        columns = [
            TableColumn(field="probabilities", title="Probabilities", width=70, formatter=probability_formatter),
            TableColumn(field="drawdowns", title="Drawdowns", width=70, formatter=drawdown_formatter)
        ]
        # display(dd_tbl)
        data_table = DataTable(source=source, columns=columns, height=220, row_headers=False)
        return data_table

    def generate_simulated_drawdown_plot(self, width=500, height=500):
        # Drawdown histogram
        p2 = figure(width=width, height=height, title='Simulated drawdowns & CDF', toolbar_location=None)
        hist, edges = np.histogram(self.drawdowns, density=True, bins=20)
        p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.3,
                fill_color="dodgerblue", line_color="midnightblue")
        p2.xaxis.axis_label = 'Drawdown'
        p2.yaxis.axis_label = 'Density'

        # Add drawdown CDF
        pp = np.linspace(0., 100., 200)
        ddp = np.percentile(self.drawdowns, pp)
        p2.extra_y_ranges['cdf'] = Range1d(0., 1.)
        p2.line(x=ddp, y=pp / 100., color="midnightblue", line_width=2, y_range_name='cdf')
        p2.add_layout(LinearAxis(y_range_name="cdf", axis_label='CDF'), 'right')
        return p2

    def generate_autocorrelation_plot(self, alpha=0.05, n=21, width=500, height=500, **kwargs):
        p = figure(width=width, height=height, toolbar_location=None, title='Autocorrelation Plot of Daily Returns',
                   **kwargs)
        N = len(self.daily_returns)
        if n > N:
            raise ValueError('Number of lags must be at least the number of points')

        val = acf(self.daily_returns, nlags=n)

        z = norm.ppf(1. - 0.5 * alpha)
        est_lim = z * np.concatenate([[0, 1 / np.sqrt(N)], np.sqrt((1 + 2. * np.cumsum(val[1:] ** 2)) / N)])
        x = range(n)

        xx = x + x[::-1] + x[:1]
        yy = est_lim[xx] * np.array([1.] * n + [-1.] * n + [1.])

        p.patch(xx, yy, color='dodgerblue', alpha=0.3)
        p.line([0, n], [0, 0], color='midnightblue')
        for ii in x:
            p.line([ii, ii], [0., val[ii]], color='midnightblue')
        p.scatter(x, val[:n], color='midnightblue')
        p.xaxis.axis_label = 'Lag'
        p.yaxis.axis_label = 'AC coefficient'

        return p

    def generate_daily_returns_plot(self, width=500, height=500):
        line_rv = np.linspace(self.chosen_rv.ppf(0.001), self.chosen_rv.ppf(0.999), 100)
        p1 = figure(width=width, height=height, title='Daily Returns & Fitted Distributions', tools=[])

        hist, edges = np.histogram(self.daily_returns, density=True, bins=20)
        p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.3, legend='daily returns',
                fill_color="dodgerblue", line_color="midnightblue")

        p1.line(line_rv, np.exp(self.chosen_rv.logpdf(line_rv)), color='midnightblue',
                legend=self.chosen_rv.dist.name, line_width=2)

        p1.xaxis.axis_label = 'Return'
        p1.yaxis.axis_label = 'Density'
        return p1


def examine_daily_distributions(returns, name, n_tr_days):
    '''
    Function to use in a notebook to organise all the output
    :param returns: Daily returns to model
    :param name: Name of the strategy
    :param n_tr_days: Number of trading days of the strategy
    :return:
    '''
    drv = DailyReturnVisualisations(name, returns, n_trading_days=n_tr_days, n_drawdown_simulations=10000)
    display(HTML('<h3>{} Daily Returns</h3>'.format(name)))
    display(HTML('{} days, best fitting distribution is {}'.format(len(returns), drv.chosen_rv.dist.name)))
    tabs = [Panel(child=drv.generate_daily_returns_plot(), title='Daily returns'),
            Panel(child=drv.generate_simulated_drawdown_plot(), title='Drawdown'),
            Panel(child=drv.generate_autocorrelation_plot(), title='Autocorrelation')]
    _ = show(column(drv.generate_summary_table(),
                    Tabs(tabs=tabs),
                    drv.generate_simulated_drawdown_table(),
                    drv.generate_stationarity_table()))