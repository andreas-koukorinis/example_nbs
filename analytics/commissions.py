from sgmtradingcore.strategies.config.configurations import BETA_CONFIG, PROD_ALGO_SPORTS_CONFIG, \
    PROD_CONFIG, DEV_CONFIG
from stratagemdataprocessing import data_api
from stratagemdataprocessing.enums.odds import Bookmakers
from stratagemdataprocessing.parsing.common.stickers import BOOKMAKER_ABBR

DEFAULT_COMMISSIONS = {
    BOOKMAKER_ABBR[Bookmakers.SBO_BET]: 0.0,

    # Commission is 5%, ignoring the Discount Rate
    BOOKMAKER_ABBR[Bookmakers.BETDAQ]: 0.05,
    BOOKMAKER_ABBR[Bookmakers.PINNACLE_SPORTS]: 0.0,
    BOOKMAKER_ABBR[Bookmakers.A3ET]: 0.0,

    # Commission is applied to the win amount on winning bets, or the lesser of the stake or potential win amount,
    # for losing bets.
    # Commission in halved if you we are posting an offer instead of matching one
    # Cricket have a different commission
    BOOKMAKER_ABBR[Bookmakers.MATCHBOOK]: 0.015,  # Only half of this for matching a bet

    # Commission is 5%, ignoring the Discount Rate which depends from volume
    BOOKMAKER_ABBR[Bookmakers.BETFAIR]: 0.05,
}


def get_current_commission(config):
    commission_dict = DEFAULT_COMMISSIONS.copy()

    trading_user_id = config['trading_user_id']
    account_ids = config['account_ids']

    curr_info = data_api.get_commission_info(trading_user_id, account_ids)

    # update values
    for key, acc_rates in curr_info.items():
        if len(set(acc_rates.values())) == 1:  # if it is the same rate for all accounts
            commission_dict[key] = acc_rates.values()[0]

    return commission_dict


def get_commission_rates_for_user(trading_user_id):
    if trading_user_id in ['default', 'test']:
        return DEFAULT_COMMISSIONS
    for conf in [BETA_CONFIG, PROD_ALGO_SPORTS_CONFIG, PROD_CONFIG,DEV_CONFIG]:
        if conf['trading_user_id'] == trading_user_id:
            break
    else:
        raise ValueError('Could not find config for trading_user_id {}'.format(trading_user_id))

    return get_current_commission(conf)


def get_net_odds(odds, bookmaker=None, is_back=True, commissions=None):
    if commissions is None:
        return odds

    bookmaker_abbr = BOOKMAKER_ABBR.get(bookmaker, None)
    rate = commissions.get(bookmaker_abbr, 0)
    if rate == 0:
        return odds

    if bookmaker == Bookmakers.MATCHBOOK:
        if is_back:
            net = 1 + (((odds - 1) * (1 - rate)) / (1 + (min(1, odds - 1) * rate)))
        else:
            net = 1 + ((odds - 1) + min(1, odds - 1) * rate) / (1 - rate)

    else:
        if is_back:
            net = ((odds - 1) * (1 - rate)) + 1
        else:
            net = ((odds - 1) / (1 - rate)) + 1

    return net


def get_gross_odds(odds, bookmaker, is_back, commissions):
    bookmaker_abbr = BOOKMAKER_ABBR[bookmaker]
    rate = commissions.get(bookmaker_abbr, 0)

    if bookmaker in (Bookmakers.BETFAIR, Bookmakers.BETDAQ):
        if is_back:
            gross = ((odds - 1) / (1 - rate)) + 1
        else:
            gross = ((odds - 1) * (1 - rate)) + 1

    elif bookmaker == Bookmakers.MATCHBOOK:
        if is_back:
            if odds >= 2:
                gross = (odds-1)*(1+rate)/(1-rate) + 1
            else:
                gross = (odds-1)/((1-rate)-(odds-1)*rate) + 1
        else:
            if odds >= 2:
                gross = (odds-1)*(1-rate) - rate + 1
            else:
                gross = (odds-1)*(1-rate)/(rate+1) + 1
    else:
        gross = odds

    return gross
