from stratagemdataprocessing import data_api
from stratagemdataprocessing.enums.markets import TennisMarkets, TennisSelections
from stratagemdataprocessing.enums.odds import Sports
from stratagemdataprocessing.parsing.common.stickers import parse_sticker
from sgmtradingcore.analytics.results.common import determine_order_outcome_pnl
import numpy as np

_tennis_outcome_cache = {}


def determine_tennis_outcome_from_api(market, params, enp_id):
    """
    Determine the outcome of a tennis bet from the scores
    """

    global _tennis_outcome_cache

    if enp_id not in _tennis_outcome_cache:
        enp_id_int = int(enp_id[3:])
        response = data_api.get_event_outcome(Sports.TENNIS, enp_id_int)
        _tennis_outcome_cache[enp_id] = response
    else:
        response = _tennis_outcome_cache[enp_id]

    if market == TennisMarkets.MATCH_ODDS:
        player_a_result = response.get(enp_id, {}).get(
            'details', {}).get('playerAResult', -1)
        outcome, n_bet = determine_tennis_outcome(
            market, params, player_a_result)
    elif market in [TennisMarkets.SET_BETTING, TennisMarkets.TOTAL_GAMES_OVER_UNDER, TennisMarkets.HANDICAP_GAMES]:
        scores = response.get(enp_id, {}).get('details', {})
        outcome, n_bet = determine_tennis_outcome_alternative_markets(
            market, params, scores)
    else:
        raise ValueError('implement more markets')

    return outcome, n_bet


def determine_tennis_outcome(market, params, player_a_result):
    """
    Determine the outcome of a tennis bet from the scores
    """

    n_bet = 1
    outcome = None
    if market == TennisMarkets.MATCH_ODDS:
        selection = params[0]
        if player_a_result == -1:
            return outcome, n_bet
        player_a_win = player_a_result == 'winner'

        if selection == TennisSelections.PLAYER_A:
            outcome = 1 if player_a_win else -1
        elif selection == TennisSelections.PLAYER_B:
            outcome = 1 if not player_a_win else -1
        else:
            raise ValueError('FT_12 bet type should be ONE or TWO')
    else:
        raise ValueError('implement more markets')

    return outcome, n_bet


def tennis_pnl_grid_match_winner(orders):
    """
    Returns a 1d array containing the pnl in case player A or player B wins the match. Only works with MATCH ODDS
    market
    """
    pnl_per_result = np.zeros(2)
    for idx, player_a_res in enumerate(['winner', 'loser']):
        for order in orders:
            _, _, _, params, _ = parse_sticker(order.sticker)
            nwin, n_bet = determine_tennis_outcome(
                TennisMarkets.MATCH_ODDS, params, player_a_res)
            outcome, pnl = determine_order_outcome_pnl(
                order, nwin, n_bet, default_if_unknown=False)
            pnl_per_result[idx] += pnl

    return pnl_per_result


def determine_tennis_outcome_alternative_markets(market, params, scores):
    """
    Determine the outcome of a tennnis bet on markets alternative from match winner from the scores
    """

    n_bet = 1
    outcome = None
    if market == TennisMarkets.SET_BETTING:
        if not scores:
            return outcome, n_bet
        else:
            sets_won = [0, 0]
            for set in ['Set1', 'Set2', 'Set3', 'Set4', 'Set5']:
                if scores[set][0] > scores[set][1]:
                    sets_won[0] += 1
                elif scores[set][1] > scores[set][0]:
                    sets_won[1] += 1

            selection_idx = params[0] - 1
            other_idx = selection_idx - 1
            if params[1] == sets_won[selection_idx] and params[2] == sets_won[other_idx]:
                outcome = 1
            else:
                outcome = -1

    elif market in [TennisMarkets.HANDICAP_GAMES, TennisMarkets.TOTAL_GAMES_OVER_UNDER]:
        if not scores:
            return outcome, n_bet
        else:
            games_won = [0, 0]
            selection_idx = params[0] - 1
            other_idx = selection_idx - 1
            for set in ['Set1', 'Set2', 'Set3', 'Set4', 'Set5']:
                games_won[0] += scores[set][0]
                games_won[1] += scores[set][1]

        if market == TennisMarkets.HANDICAP_GAMES:
            handicap = params[1]
            games_won[selection_idx] += handicap
            if games_won[selection_idx] > games_won[other_idx]:
                outcome = 1
            else:
                outcome = -1

        elif market == TennisMarkets.TOTAL_GAMES_OVER_UNDER:
            actual_tot_games = sum(games_won)
            tot_games_line = params[1]
            over = actual_tot_games > tot_games_line
            if params[0] == 3:
                if over:
                    outcome = 1
                else:
                    outcome = -1
            elif params[0] == 4:
                if over:
                    outcome = -1
                else:
                    outcome = 1

    else:
        raise ValueError('Market not implemented')

    return outcome, n_bet
