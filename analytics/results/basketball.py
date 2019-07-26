from stratagemdataprocessing import data_api
from stratagemdataprocessing.enums.markets import BasketballMarkets, BasketballSelections

from stratagemdataprocessing.enums.odds import Sports

def determine_basketball_outcome_from_api(market, params, enp_id):
    """
    Determine the outcome of a basketball bet from the scores

    Team A is always home for enet data
    """

    n_bet = 1
    outcome = None
    if market == BasketballMarkets.FULL_TIME_POINT_SPREAD:
        enp_id_int = int(enp_id[3:])
        selection = params[0]
        handicap = params[1]
        response = data_api.get_event_outcome(Sports.BASKETBALL, enp_id_int)

        score_home = response.get(enp_id, {}).get('details', {}).get('teamAResult', -1)
        score_away = response.get(enp_id, {}).get('details', {}).get('teamBResult', -1)


        if selection == BasketballSelections.HOME_TEAM:
            hc_score = score_home + handicap
            if hc_score == score_away:
                outcome = 0
            elif hc_score > score_away:
                outcome = 1
            else:
                outcome = -1

        elif selection == BasketballSelections.AWAY_TEAM:
            hc_score = score_away + handicap
            if hc_score == score_home:
                outcome = 0
            elif hc_score > score_home:
                outcome = 1
            else:
                outcome = -1

        else:
            raise ValueError('FTPS bet should be ONE or TWO')

    elif market == BasketballMarkets.FULL_TIME_MONEYLINE:
        enp_id_int = int(enp_id[3:])
        selection = params[0]
        response = data_api.get_event_outcome(Sports.BASKETBALL, enp_id_int)

        score_home = response.get(enp_id, {}).get('details', {}).get('teamAResult', -1)
        score_away = response.get(enp_id, {}).get('details', {}).get('teamBResult', -1)

        if selection == BasketballSelections.HOME_TEAM:
            if score_home == score_away:
                outcome = 0
            elif score_home > score_away:
                outcome = 1
            else:
                outcome = -1

        elif selection == BasketballSelections.AWAY_TEAM:
            if score_away == score_home:
                outcome = 0
            elif score_away > score_home:
                outcome = 1
            else:
                outcome = -1

        else:
            raise ValueError('selection should be ONE or TWO')
    elif market == BasketballMarkets.FULL_TIME_TOTAL_POINTS:
        enp_id_int = int(enp_id[3:])
        selection = params[0]
        handicap = params[1]
        response = data_api.get_event_outcome(Sports.BASKETBALL, enp_id_int)

        score_home = response.get(enp_id, {}).get('details', {}).get('teamAResult', -1)
        score_away = response.get(enp_id, {}).get('details', {}).get('teamBResult', -1)
        score_total = score_home + score_away

        if selection == BasketballSelections.OVER:
            if score_total == handicap:
                outcome = 0
            elif score_total > handicap:
                outcome = 1
            else:
                outcome = -1

        elif selection == BasketballSelections.UNDER:
            if score_total == handicap:
                outcome = 0
            elif score_total < handicap:
                outcome = 1
            else:
                outcome = -1

        else:
            raise ValueError('FTTP bet should be OVER or UNDER')
    else:
        raise ValueError('implement more markets')

    return outcome, n_bet
