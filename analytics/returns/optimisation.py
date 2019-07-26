import traceback

import cvxopt as opt
import numpy as np
from cvxopt import solvers, blas

from sgmtradingcore.analytics.returns.util import nearestPD


def optimal_markowitz_portfolio(r, C, logger, constraints):
    """
    :param r: (array_like) returns
    :param C: (array_like) covariance matrix
    :param constraints: (dict) with keys
        - min_wi: minimum value for a weight in the optimisation
        - max_wi: maximum value for a weight in the optimisation
    :return: efficient frontier of the portfolio as returns, variances, weights

    References
    ----------
    https://blog.quantopian.com/markowitz-portfolio-optimization-2/

    """

    n = len(r)
    logger.info('Markowitz optimisation for {n} competitions'.format(n=n))
    r = np.asmatrix(r)
    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    C = np.nan_to_num(C)  # TODO fillna with 0 for covariance matrix is not correct and should be done better
    # One way we could do it is to find the worst case scenario given our target.
    # Close form solution available if the matrix follows a specific pattern (cf

    S = opt.matrix(C)
    pbar = opt.matrix(np.mean(r, axis=1))

    # Create constraint matrices
    # Constraint Gw <= h

    logger.info('Adding constraint matrices for {} method'.format(constraints['type']))
    if constraints['type'] == 'min_max':
        uni_wi = 1. / len(C)
        min_wi = constraints['min_coef'] * uni_wi
        max_wi = constraints['max_coef'] * uni_wi
        logger.info('Weights boundaries in optimisation are [{}, {}]'.format(min_wi, max_wi))
        G_ = np.concatenate((-np.eye(n), np.eye(n)), axis=0)
        G = opt.matrix(G_)  # negative n x n identity matrix
        h_min = - min_wi * np.ones((n, 1))
        h_max = + max_wi * np.ones((n, 1))
        h_ = np.concatenate((h_min, h_max), axis=0)
        h = opt.matrix(h_)
    elif constraints['type'] == 'smooth':
        raise NotImplementedError('TODO smooth contraint !')
    else:
        raise ValueError('Constraint type {} is not understood by markowitz implementation'.
                         format(constraints['type']))
    # Constraint Aw = b
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    weights = []
    no_fails = 0
    no_success = 0
    err_log = []
    for mu in mus:
        try:
            P = mu * S
            try:
                solver_result = solvers.qp(P, -pbar, G, h, A, b)['x']
            except ValueError:
                S = nearestPD(np.array(S), alternative=True)
                S = opt.matrix(S)
                P = mu * S
                solver_result = solvers.qp(P, -pbar, G, h, A, b)['x']
            weights.append(solver_result)
            no_success += 1
        except:
            err_log.append(str(traceback.format_exc()))
            no_fails += 1
    if no_fails > 0:
        logger.warning('{}/{} optimisations failed !'.format(no_fails, no_fails + no_success))
    if no_fails > no_success:
        logger.error(err_log[0])
        raise Exception('Too many fails for the optimisation ! Please debug')

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in weights]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in weights]

    return returns, risks, [np.asarray(por) for por in weights]