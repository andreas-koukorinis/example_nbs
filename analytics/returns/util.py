import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML
from matplotlib import colors
from numpy import linalg as la

from sgmtradingcore.analytics.metrics import _format_df_for_capital_metrics
from sgmtradingcore.analytics.metrics import flat_capital_traces
from sgmtradingcore.strategies.config.configurations import TRADING_USER_MAP
from stratagemdataprocessing.enums.markets import Selections
from stratagemdataprocessing.parsing.common.stickers import parse_sticker, \
    MARKET_ABBR
from stratagemdataprocessing.util.dateutils import parse_date_if_necessary


class StrategySettings(object):
    """
    Class util to identify backtest runs settings
    """
    def __init__(self, strategy_name, strategy_desc, strategy_code,
                 mnemonic, trading_user_id):
        self.strategy_name = strategy_name
        self.strategy_desc = strategy_desc
        self.strategy_code = strategy_code
        self.mnemonic = mnemonic
        self.trading_user_id = trading_user_id

    def __str__(self):
        str_strategy_code = ('' if self.strategy_code is None else
                             '-' + self.strategy_code)
        str_trading_user_id = TRADING_USER_MAP[self.trading_user_id]

        str_ = '[{sn}] {sd}{sc} ({tui}) {mn}'.format(sn=self.strategy_name,
                                                     sd=self.strategy_desc,
                                                     sc=str_strategy_code,
                                                     tui=str_trading_user_id,
                                                     mn=self.mnemonic)
        return str_

    @property
    def key(self):
        """
        The returned key is unique and is enough to identify a 
            backtest results
        Any change in one of the parameters in the key would change 
            the backtest result
        
        Returns
        -------
        key: (tuple)

        """
        return (self.strategy_name, self.strategy_desc,
                self.strategy_code, self.mnemonic,
                self.trading_user_id)

    @staticmethod
    def init_from_key(key):
        return StrategySettings(key[0], key[1], key[2], key[3], key[4])


def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    """
    Function to return the background color gradient by specifying the
     minimum and maximum values m and M
    :param s: DataFrame Style
    :param m: minimum value the data can take
    :param M: maximum value the data can take
    :param cmap: sns
    :return:
    """
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]


def display_df_colors(df, caption, cmap=None):
    """
    Function to display the DataFrame with background highlighted with
     colors

    :param df: DataFrame to display
    :param caption: Title of the table/DataFrame to display
    :param cmap: color map
    :return: (None) display the DataFrame in a nice colored format
    """
    if cmap is None:
        cmap = sns.diverging_palette(5, 250, as_cmap=True)
    display(df.style.background_gradient(cmap=cmap).
            set_properties(**{'max-width': '80px',
                              'font-size': '10pt'}).
            set_caption(caption).set_precision(4))


def display_df_style(df, caption, m, M, cmap=None):
    """
    Function to display the DataFrame with background highlighted with
     colors

    :param df: DataFrame to display
    :param caption: Title of the table/DataFrame to display
    :param m: minimum value the data can take
    :param M: maximum value the data can take
    :param cmap: color map
    :return: (None) display the DataFrame in a nice colored format
    """
    if cmap is None:
        cmap = sns.diverging_palette(5, 250, as_cmap=True)
    display(df.style.apply(background_gradient, cmap=cmap, m=m, M=M).
            set_properties(**{'max-width': '80px',
                              'font-size': '10pt'}).
            set_caption(caption).set_precision(4))



def get_traces(df, grp_by, ret_aggr_by='date'):
    if grp_by is None:
        return flat_capital_traces(
            df, groupby=None, return_agg_by=ret_aggr_by).reset_index()
    traces = df.groupby(grp_by).apply(
        flat_capital_traces, return_agg_by=ret_aggr_by).reset_index()
    return traces


def get_aggr_returns(_df, ret_grp_by):
    return _format_df_for_capital_metrics(_df, None)[0].groupby(
        [ret_grp_by]).sum()['return'].T


def f_order_to_row(sport, order):
    return {'fixture_id': int(order['event_id'][3:]),
            'instruction_id': order['instruction_id'],
            'sticker': order['sticker'],
            'market_id': order['market_id'],
            'selection_id': order['selection'],
            'handicap': order.get('handicap', np.nan),
            'is_back': (order['bet_side'] == 'back'),
            'bookmaker': order['execution_details']['bookmaker'],
            'placed_odds': order['price'],
            'matched_odds': order['average_price_matched'],
            'placed_size': order['size'],
            'matched_size': order['size_matched'],
            'pnl': order['pnl'],
            'status': order['status_str'],
            'dt': order['placed_time'],
            'capital_received': order['details'].get('capital_received', -1),
            'placed_date': order['date_day'],
            'trade_id': order['trade_id'],
            'extra_details': order['details']}


def f_instruction_to_row(sport, instr, extra_info=None):
    '''
    :param extra_info: (dict) extra information to take from raw instructions,
        e.g. {'timestamp': 'details.signals.0.timestamp'}
    '''
    _, (_, event_id), mkt_id, params, _ = parse_sticker(instr['sticker'])
    sel_id = params[0]
    hc = np.nan if len(params) == 1 else params[1]

    instr_dict = {
        'fixture_id': int(event_id[3:]),
        'average_price_matched': instr['average_price_matched'],
        'capital_received': instr['details'].get('capital_received', -1),
        'sticker': instr['sticker'],
        'market': MARKET_ABBR[sport][mkt_id],
        'selection': Selections.to_str(sel_id),
        'hc': hc,
        'is_back': (instr['bet_side'] == 'back'),
        'instruction_id': instr['id'],
        'selection_id': sel_id,
        'market_id': mkt_id,
        'handicap': hc,
        'placed_time': parse_date_if_necessary(instr['placed_time'],
                                               to_utc=True),
        'size_wanted': instr['size'],
        'size_matched': instr['size_matched'],
        'strategy': instr['strategy'],
        'strategy_descr': instr['strategy_descr'],
        'trade_id': instr['trade_id'],
    }

    if extra_info is not None:
        # get specified additional information from instruction
        info_dict = {}  # initialise
        for label, i in extra_info.iteritems():
            i_keys = i.split('.')
            info = instr.copy()  # initialise
            for i_key in i_keys:
                try:
                    i_key = int(i_key)  # convert to int if possible
                except:
                    pass
                info = info[i_key]  # update
            info_dict[label] = info

        instr_dict.update(info_dict)
    return instr_dict


def custom_notebook_display(var):
    """
    Function to display things in a notebook.
    
    Parameters
    ----------
    var: (HTML or DataFrame or basestring)

    Returns
    -------
    Nothing, it displays var in a notebook
    """
    if isinstance(var, HTML):
        display(var)
    elif isinstance(var, pd.DataFrame):
        display(np.round(var, 3))
    elif isinstance(var, basestring):
        display(HTML('<p>{}</p>'.format(var)))
    else:
        raise ValueError('Expected HTML or DataFrame. Received {}'.
                         format(type(var)))


def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q * xdiag * Q.T


def _getPs(A, W=None):
    W05 = np.matrix(W ** .5)
    return W05.I * _getAplus(W05 * A * W05) * W05.I


def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)


def nearestPD(A, alternative=False):
    if alternative:
        return _nearestPD_approx(A, nit=50)
    else:
        return _nearestPD_best(A)


def _nearestPD_approx(A, nit=10):
    n = A.shape[0]
    W = np.identity(n)
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk


def _nearestPD_best(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    if isPD(A):
        return A

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False