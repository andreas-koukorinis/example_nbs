
from sgmtradingcore.core.trading_types import OrderOutcome, OrderStatus


def determine_order_outcome_pnl(order, n_win, n_bet, default_if_unknown=False):
    size = order_size(order, default_if_unknown)
    odds = order_odds(order, default_if_unknown)
    if n_win > 0:
        if n_win < n_bet:
            outcome = OrderOutcome.HALF_WIN
            pnl = 0.5 * size * (odds - 1.0)
        else:
            outcome = OrderOutcome.WIN
            pnl = size * (odds - 1.0)
    elif n_win < 0:
        if n_win > -n_bet:
            outcome = OrderOutcome.HALF_LOSS
            pnl = size * -0.5
        else:
            outcome = OrderOutcome.LOSS
            pnl = size * -1.0
    else:
        pnl = 0.0
        outcome = OrderOutcome.PUSH

    # Reverse the pnl and outcome for lay bets
    if not order.is_back:
        pnl *= -1.0

        if outcome == OrderOutcome.WIN:
            outcome = OrderOutcome.LOSS
        elif outcome == OrderOutcome.HALF_WIN:
            outcome = OrderOutcome.HALF_LOSS
        elif outcome == OrderOutcome.HALF_LOSS:
            outcome = OrderOutcome.HALF_WIN
        elif outcome == OrderOutcome.LOSS:
            outcome = OrderOutcome.WIN

    return outcome, pnl


def order_size(order, default_if_unknown=False):
    size = order.matched_amount
    if default_if_unknown and order.matched_amount == 0. and order.status == OrderStatus.UNKNOWN:
        size = order.size
    return size


def order_odds(order, default_if_unknown=False):
    odds = order.matched_odds
    if default_if_unknown and order.matched_odds == 0. and order.status == OrderStatus.UNKNOWN:
        odds = order.price
    return odds
