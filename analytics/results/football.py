from itertools import product
from math import floor, ceil
from stratagemdataprocessing import data_api
from stratagemdataprocessing.enums.odds import Sports
from stratagemdataprocessing.enums.markets import Markets, Selections
from stratagemdataprocessing.parsing.common.stickers import parse_sticker
from sgmtradingcore.analytics.results.common import determine_order_outcome_pnl, order_size
import numpy as np


def get_fball_scores(market, fixture_details):
    if Markets.is_goal_mkt(market):
        final_score = (fixture_details['team1FTG'], fixture_details['team2FTG'])
        half_time_score = (fixture_details['team1HTG'], fixture_details['team2HTG'])
    elif Markets.is_corner_mkt(market):
        if not ('team1FTC' in fixture_details and 'team2FTC' in fixture_details):
            raise ValueError('Corner score not found for corner market.')
        final_score = (fixture_details['team1FTC'], fixture_details['team2FTC'])
        half_time_score = (
            (fixture_details['team1HTC'], fixture_details['team2HTC'])
            if ('team1HTC' in fixture_details and 'team2HTC' in fixture_details)
            else None)
    else:
        raise ValueError('Unsupported market ({}).'.format(market))
    return final_score, half_time_score


def determine_football_outcome(market, params, final_score, half_time_score=None):
    """
      Determine the outcome of a football bet from the score
      Notes:
      - Use final_score for FULL TIME and half_time_score for HALF TIME markets
      - If market is for corners, final_score and half_time_score should also be for corners
    :param market:
    :param params:
    :param final_score: for full time markets outcome
    :param half_time_score: for half time markets outcome
    :return:
    """
    # Sanity checks
    if Markets.is_halftime(market):
        if half_time_score is None:
            raise ValueError('Half time markets: %s need half_time_score input' % market)
        score_home = half_time_score[0]
        score_away = half_time_score[1]

    elif Markets.is_fulltime(market):
        if final_score is None:
            raise ValueError('Full time markets: %s need final_score input' % market)

        score_home = final_score[0]
        score_away = final_score[1]

    else:
        raise ValueError('market ({}) not recognised as full-time or half-time'
                         .format(market))

    outcome = 0
    n_bet = 1
    if Markets.is_1X2(market, include_double_selection=False):
        selection = params[0]
        if selection == Selections.HOST_WON:
            outcome = 1 if score_home > score_away else -1
        elif selection == Selections.GUEST_WON:
            outcome = 1 if score_away > score_home else -1
        elif selection == Selections.DRAW:
            outcome = 1 if score_away == score_home else -1
        else:
            raise ValueError('FT_1X2 bet type should be ONE, DRAW, or TWO')
    elif Markets.is_asian_hc_market(market):
        selection = params[0]
        handicap = params[1]
        if (handicap % 1) in [0.25, 0.75]:
            handicap = [floor(handicap * 2) / 2, ceil(handicap * 2) / 2]
        else:
            handicap = [handicap]

        n_bet = len(handicap)

        for hc in handicap:
            if selection == Selections.HOST_WON:
                sup = score_home - score_away + hc  # Handicap is team1 specific
                if sup > 0:
                    outcome += 1
                elif sup < 0:
                    outcome += -1
            elif selection == Selections.GUEST_WON:
                sup = score_home - score_away - hc  # Handicap is team2 specific
                if sup < 0:
                    outcome += 1
                elif sup > 0:
                    outcome += -1
            else:
                raise ValueError('FT_AH_GOALS/CORNERS bet type should be ONE or TWO')
    elif Markets.is_overunder_hc_market(market):
        selection = params[0]
        handicap = params[1]
        if (handicap % 1) in [0.25, 0.75]:
            handicap = [floor(handicap * 2) / 2, ceil(handicap * 2) / 2]
        else:
            handicap = [handicap]

        n_bet = len(handicap)

        for hc in handicap:
            tg = score_home + score_away - hc
            if selection == Selections.OVER:
                if tg > 0:
                    outcome += 1
                elif tg < 0:
                    outcome += -1
            elif selection == Selections.UNDER:
                if tg < 0:
                    outcome += 1
                elif tg > 0:
                    outcome += -1
            else:
                raise ValueError('FT_OU_GOALS/CORNERS bet type should be OVER or UNDER')
    elif Markets.is_correct_score(market):
        bet_home_score = params[0]
        bet_away_score = params[1]
        if (bet_home_score == score_home) and (bet_away_score == score_away):
            outcome = 1
        else:
            outcome = -1
    else:
        raise ValueError('Implement more markets: %s' % market)

    return outcome, n_bet


#  returns a 1-dimensional array
def football_pnl_grid(orders, normalize_by_stake=False, max_evts=7, default_if_unknown=False, reshape=False):
    '''
    :param orders: can be for (e.g.) goal or corner markets
    :param max_evts: the maximum number of events (e.g. goals or corners) or a team handled by the pnl grid
    '''
    pnl_per_score = []
    aggregate_bet_notional = 1

    if normalize_by_stake:
        aggregate_bet_notional = sum(order_size(b, default_if_unknown) for b in orders)

    for team1_score, team2_score in product(range(max_evts + 1), range(max_evts + 1)):
        pnl_for_score = 0
        for order in orders:
            _, market_scope, market, params, _ = parse_sticker(order.sticker)
            n_win, n_bet = determine_football_outcome(market, params, (team1_score, team2_score))
            outcome, pnl = determine_order_outcome_pnl(order, n_win, n_bet, default_if_unknown=default_if_unknown)
            if normalize_by_stake:
                pnl /= float(aggregate_bet_notional)
            pnl_for_score += pnl
        pnl_per_score.append(pnl_for_score)

    if reshape:
        return np.array(pnl_per_score).reshape(max_evts + 1, max_evts + 1)
    return np.array(pnl_per_score)


def football_max_loss(bet_infos):
    pnl_per_score = football_pnl_grid(bet_infos)
    # the "max" loss is the min negative pnl or 0 if the pnl is positive
    return min(0, min(pnl_per_score))


def determine_football_outcome_from_api(market, params, event_id):
    """
    Determine the outcome of a basketball bet from the scores

    Team A is always home for enet data
    """
    gsm_id = int(event_id[3:])
    response = data_api.get_event_outcome(Sports.FOOTBALL, gsm_id)
    details = response[event_id]['details']
    final_score, half_time_score = get_fball_scores(market=market, fixture_details=details)
    outcome, n_bet = determine_football_outcome(market, params, final_score, half_time_score=half_time_score)
    return outcome, n_bet
