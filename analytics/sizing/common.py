import numpy as np
from scipy.optimize import minimize_scalar


########################################################
# This file contains different sizing algorithms to use in strategies and they are themselves sport independent
# Inputs to this will be sport dependent
# Example usage in a strategy
# Note: we would do the base sizing at the beginning of a period (or somewhat infrequently)
# but size_performance should be called upon every bet being indicated
# events = fn_to_get_eligible_upcoming_events(<some timeframe>)
# base_size_1event = size_max_loss(-1000, normalized_max_loss_grid, 1, expected_bets_per_event=1)
# base_size_1d = size_max_loss(-1000, normalized_max_loss_grid, <max events in one day>, expected_bets_per_event=1)
# base_size_1w = size_max_loss(-1000, normalized_max_loss_grid, <max events in one week>, expected_bets_per_event=1)
# base_size_kelly = base_size_1event * size_kelly()
# base_size = min(base_size_1event, base_size_1d, base_size_1w, base_size_kelly)
# performance_size_factor = size_performance(recent_bet_infos, recent_return, sharpe)
# final_size = performance_size_factor * base_size
# ...
# execute_size(final_size)
# Note: strategy must still have an if statement checking if realized loss is greater than max_loss_threshold


#Usage: use football_max_loss_grid(bet_infos, normalize_by_stake=True) to get normalized_max_loss_grid
#       or pass in your own expected max_loss_grid
#Notes:
# -- returns GBP stake amount
# -- does not work for Tennis yet b/c there is no equivalent of football_max_loss_grid()
# -- still need to address the "trades" vs "bets" but in theory, it depends on what you pass as the max_loss_grid
def size_max_loss(max_loss, normalized_max_loss_grid, number_of_events=1, expected_bets_per_event=1):
    if max_loss > 0:
        raise ValueError('size_max_loss expect max_loss input < 0')
    if number_of_events <= 0:
        return 0.
    number_of_bets = expected_bets_per_event*number_of_events
    normalized_max_loss = min(0.0, min(normalized_max_loss_grid))
    max_loss_per_bet = ((float(max_loss) * -1) / float(normalized_max_loss)) / float(number_of_bets)
    return max_loss_per_bet * -1


def size_max_loss_binned_trades(max_loss, trades_per_bin, bin_scales):
    '''
    Function to size a number of trades that are scaled according to which bin they fit into

    :param max_loss: Maximum acceptable worst case loss for all trades
    :param trades_per_bin: Number of trades in each bin
    :param bin_scales: Scale for a trade in each bin
    :return: A base size to apply scales to
    '''
    if max_loss > 0:
        raise ValueError('size_max_loss_binned_trades expect max_loss input < 0')
    if len(trades_per_bin) == 0 or sum(trades_per_bin) == 0.:
        return 0.
    if any([s < 0 for s in bin_scales]):
        raise ValueError('size_max_loss_binned_trades expects bin scales to be >= 0')
    if sum(bin_scales) == 0.:
        return 0.
    total_stake = sum([n * s for n, s in zip(trades_per_bin, bin_scales)])
    return -float(max_loss) / float(total_stake)


def size_kelly(weighted_avg_odds, p, fraction=1.):
    if not weighted_avg_odds or not p:
        return 0.
    kelly_fraction = (weighted_avg_odds * p - 1.) / (weighted_avg_odds - 1.)
    if kelly_fraction <= 0.:
        return 0.
    return fraction * kelly_fraction


def size_kelly_from_returns(returns, fraction=1.):
    '''
    Function to estimate the fraction to stake in order to maximize log utility
    based on historical returns
    :param returns: Array-like - assumed to be returns from unit liability
    :param fraction: Fraction of Kelly stake to use
    :return: Proportion of capital to stake
    '''
    returns = np.array(returns)
    if len(returns) < 10:
        # Insufficient historical returns
        return 0.
    if returns.mean() < 0.:
        # No edge
        return 0.

    def f_log_utility(alpha):
        return -np.mean(np.log(alpha * returns + 1))
    opt = minimize_scalar(f_log_utility, bounds=(0.01, 0.99), method='bounded')
    kelly_fraction = opt['x']
    return fraction * kelly_fraction


# Usage: performance_size_factor = size_performance(recent_bet_infos, recent_return, sharpe)
# Returns: a factor to size by (1.2x means go 20% bigger than whatever base size you have)
# Notes:
# -- could add drawdown and vol level as well as arguments
# -- strategy gets recent_return, recent_sharpe from RiskProvider
LOWER_SHARPE_BOUND = -3
UPPER_SHARPE_BOUND = 3
SIZE_AT_LOWER_SHARPE_BOUND = 0.25
SIZE_AT_UPPER_SHARPE_BOUND = 1.5
def size_performance(recent_bet_infos, recent_sharpe):
    if len(recent_bet_infos) < 100:
        return 1
    if recent_sharpe < -3:
        return 0
    if recent_sharpe > UPPER_SHARPE_BOUND:
        return SIZE_AT_UPPER_SHARPE_BOUND

    factor = (recent_sharpe - LOWER_SHARPE_BOUND) / (UPPER_SHARPE_BOUND - LOWER_SHARPE_BOUND)
    return factor * (SIZE_AT_UPPER_SHARPE_BOUND - SIZE_AT_LOWER_SHARPE_BOUND)

