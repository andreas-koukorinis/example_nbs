import numpy as np

class Scoring(object):
    """
    This class provides several scoring functions
    """
    @staticmethod
    def calculate_score_total_return(metrics):
        """
        Override to calculate the score using one or more of the standard metrics
        """
        return metrics['total_pnl']

    @staticmethod
    def _calculate_points_linear(val, bounds, point_scale):
        """
        Simple linear method of determining "score points" from bounds
        """
        factor = (val - bounds[0]) / (bounds[1] - bounds[0])
        factor = max(factor, 0)
        factor = min(factor, 1)
        return factor * (point_scale[1] - point_scale[0])

    @staticmethod
    def _calculate_points_general(value, good_value, bad_value, point_scale, truncate_left=True,
                                  truncate_right=True, control=0.):
        if truncate_left:
            value = np.maximum(value, bad_value)
        if truncate_right:
            value = np.minimum(value, good_value)
        scaled_value = (value - bad_value) / (good_value - bad_value)
        if control == 0.:
            return point_scale * scaled_value
        else:
            return point_scale * (np.exp(control * scaled_value) - 1) / (np.exp(control) - 1)

    @staticmethod
    def calculate_score_100(metrics, return_bound=[-0.05, 0.20], min_trades=100,
                            return_weight=30, sharpe_weight=30, drawdown_weight=40):
        """
        Simple metholodogy: 50 points on return, 25 on drawdown, 25 points on sharpe; linearly scaled
        If there are less than 100 bets, it's only possible to achieve 50 points (automatically deemphasizing low betting runs)
        """

        if metrics['n_trades'] < min_trades:
            return 0.

        cum_return = metrics['cum_return']
        sharpe_ratio = metrics['sharpe_ratio']
        drawdown = metrics['maximum_drawdown']

        return_score = Scoring._calculate_points_linear(cum_return, return_bound, [0., return_weight])

        sharpe_score = 0
        drawdown_score = 0
        if metrics['n_trades'] > min_trades:
            sharpe_bound = [-1., 3.]
            sharpe_score = Scoring._calculate_points_linear(sharpe_ratio, sharpe_bound, [0., sharpe_weight])

            drawdown_bound = [-0.15, 0.]
            drawdown_score = Scoring._calculate_points_linear(drawdown, drawdown_bound, [0., drawdown_weight])
        return return_score + sharpe_score + drawdown_score

    @staticmethod
    def calculate_general_score(metrics, min_trades=100, **kwargs):
        """
        kwargs define the metrics defined as <metric_name> = (args, kw) which are passed to _calculate_points_general
        """
        if metrics['n_trades'] < min_trades:
            return -np.inf

        arguments = {'maximum_drawdown': ([-0.05, -0.2, 25.],
                                          {'truncate_left': False, 'truncate_right': True, 'control': -2.0}),
                     'cum_return': ([0.2, 0.0, 25.],
                                         {'truncate_left': False, 'truncate_right': True, 'control': -2.0}),
                     'sharpe_ratio': ([3., 0., 50.],
                              {'truncate_left': True, 'truncate_right': True, 'control': 0.0})}

        if len(kwargs) > 0:
            arguments = kwargs

        return sum([Scoring._calculate_points_general(metrics[metric_name], *args, **kw)
                    for metric_name, (args, kw) in arguments.iteritems()])

    @staticmethod
    def calculate_score_sharpe_drawdown(metrics):
        """
        Simple metholodogy: 35 on drawdown, 65 points on sharpe; linearly scaled
        If there are less than 100 bets, score = 0
        """

        sharpe_ratio = metrics['sharpe_ratio']
        drawdown = metrics['maximum_drawdown']

        sharpe_score = 0
        drawdown_score = 0
        if metrics['n_trades'] > 100:
            sharpe_bound = [-1., 3.]
            sharpe_point_scale = [0., 65.]
            sharpe_score = Scoring._calculate_points_linear(sharpe_ratio, sharpe_bound, sharpe_point_scale)

            drawdown_bound = [-0.15, 0.]
            drawdown_point_scale = [0., 35.]
            drawdown_score = Scoring._calculate_points_linear(drawdown, drawdown_bound, drawdown_point_scale)
        return sharpe_score + drawdown_score
