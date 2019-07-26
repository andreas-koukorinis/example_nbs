import logging
from sgmtradingcore.analytics.results.common import determine_order_outcome_pnl
from sgmtradingcore.analytics.results.football import determine_football_outcome
from sgmtradingcore.analytics.results.tennis import determine_tennis_outcome, \
    determine_tennis_outcome_alternative_markets
from sgmtradingcore.analytics.results.basketball import determine_basketball_outcome_from_api
from stratagemdataprocessing import data_api
from stratagemdataprocessing.data_api import get_settled_orders
from stratagemdataprocessing.enums.markets import Markets, TennisMarkets, BasketballMarkets
from stratagemdataprocessing.enums.odds import Sports
from stratagemdataprocessing.parsing.common.stickers import MarketScopes
from collections import defaultdict
from sgmtradingcore.execution.monitoring import json_to_bet_info
from stratagemdataprocessing.parsing.common.stickers import parse_sticker, extract_sport

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - ''%(levelname)s - %(message)s')

TRADING_USERS = {'algosports': '562f5bef497aee1c22000001',
                 'stratagem': '54da2b5fd47e6bff0dade9b4',
                 'dev': '552f6a9139fdca41ca28b01a'}

_FLUSH_NUM_ORDERS = 200


# TODO this is completely untested
class PnlChecker(object):
    def __init__(self, start_dt, end_dt, strategy_names, providers, sport):
        self.orders_by_provider = defaultdict(list)
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.sport = sport
        self.strategy_names = strategy_names
        self.providers = providers
        self.flushed = False

    def run(self):
        for trading_user_id in TRADING_USERS.values():
            for strategy_name in self.strategy_names:
                orders = get_settled_orders(
                    trading_user_id, self.start_dt, self.end_dt, strategy_name)

                for order in orders:
                    try:
                        bet_info = json_to_bet_info(order)
                    except KeyError:
                        logging.error("Missing Sticker #Order %s\n" % order)
                        continue
                    sport, (scope, event_id), market, params, bm = parse_sticker(bet_info.sticker)
                    exec_details = bet_info.execution_details
                    provider = exec_details['provider']
                    if provider in self.providers:
                        bookmaker = exec_details['bookmaker']
                        sticker = bet_info.sticker
                        if len(sticker) == 0:
                            logging.error("Missing Sticker #Order %s\n" % order)
                            continue

                        if scope == MarketScopes.EVENT:
                            # just event scopes for now
                            current_orders = self.orders_by_provider[event_id, provider, bookmaker]
                            current_orders.append(bet_info)

                            if len(self.orders_by_provider) >= _FLUSH_NUM_ORDERS:
                                # Calculate outcomes for all event_ids
                                event_ids = self._get_event_ids()
                                response = data_api.get_event_outcome(sport, event_ids)
                                self.report_outcome_for_events(response)
                                self.report_missing_fx_rate()
                                # flush dictionary
                                self.orders_by_provider.clear()
                                self.flushed = True

        if self.orders_by_provider:
            # Calculate outcomes for all event_ids
            event_ids = self._get_event_ids()
            response = data_api.get_event_outcome(extract_sport(bet_info.sticker), event_ids)
            self.report_outcome_for_events(response)
            self.report_missing_fx_rate()
        elif not self.flushed:
            logging.info("No orders found")

    def report_outcome_for_events(self, response):
        if self.sport == Sports.FOOTBALL:
            self.report_outcome_for_football(response)
        elif self.sport == Sports.TENNIS:
            self.report_outcome_for_tennis(response)
        elif self.sport == Sports.BASKETBALL:
            self.report_outcome_for_basketball()

    def _get_event_ids(self):
        event_ids = []
        for key in self.orders_by_provider.keys():
            event_id = key[0]
            if self.sport == Sports.FOOTBALL:
                id = event_id.split("GSM")
            elif self.sport == Sports.TENNIS:
                id = event_id.split("ENP")
            elif self.sport == Sports.BASKETBALL:
                id = event_id.split("ENP")
            else:
                raise ValueError('Sport %s not supported', self.sport)

            try:
                id = int(id[1])
                event_ids.append(id)
            except IndexError:
                logging.debug("No supported id %s" % event_id)
                continue

        return event_ids

    def report_outcome_for_tennis(self, response):
        for key, orders in self.orders_by_provider.iteritems():
            for bet_info in orders:
                sport, (_, event_id), market, params, bm = parse_sticker(bet_info.sticker)
                enp_id = key[0]

                try:
                    details = response[enp_id]['details']
                except KeyError:
                    logging.debug("Manual #Order %s" % bet_info)
                    continue

                player_a_result = details.get('playerAResult', -1)

                if market == TennisMarkets.MATCH_ODDS:
                    n_win, n_bet = determine_tennis_outcome(market, params, player_a_result)
                    report_bet_outcome_pnl(n_win, n_bet, bet_info)
                elif market in [TennisMarkets.SET_BETTING, TennisMarkets.TOTAL_GAMES_OVER_UNDER, TennisMarkets.HANDICAP_GAMES]:
                    scores = response.get(enp_id, {}).get('details', {})
                    n_win, n_bet = determine_tennis_outcome_alternative_markets(market, params, scores)
                    report_bet_outcome_pnl(n_win, n_bet, bet_info)
                else:
                    logging.error("Cannot determine market pnl #Order %s\n" % bet_info)
                    continue

    def report_outcome_for_football(self, response):
        for key, orders in self.orders_by_provider.iteritems():
            for bet_info in orders:
                sport, (_, event_id), market, params, bm = parse_sticker(bet_info.sticker)
                gsm_id = key[0]

                try:
                    details = response[gsm_id]['details']
                except KeyError:
                    logging.debug("Manual #Order %s" % bet_info)
                    continue

                last_score_ft = (details['team1FTG'], details['team2FTG'])
                last_score_ht = (details['team1HTG'], details['team2HTG'])

                if market in (Markets.FULL_TIME_ASIAN_HANDICAP_GOALS, Markets.FULL_TIME_OVER_UNDER_GOALS,
                              Markets.FULL_TIME_CORRECT_SCORE, Markets.FULL_TIME_1X2):
                    n_win, n_bet = determine_football_outcome(market, params, last_score_ft)
                elif market in (Markets.HALF_TIME_ASIAN_HANDICAP_GOALS, Markets.HALF_TIME_OVER_UNDER_GOALS,
                                Markets.HALF_TIME_CORRECT_SCORE, Markets.HALF_TIME_1X2):
                    n_win, n_bet = determine_football_outcome(market, params, last_score_ft, last_score_ht)
                else:
                    logging.error("Cannot determine market pnl #Order %s\n" % bet_info)
                    continue

                report_bet_outcome_pnl(n_win, n_bet, bet_info)

    def report_outcome_for_basketball(self):
        for key, orders in self.orders_by_provider.iteritems():
            for bet_info in orders:
                sport, (_, event_id), market, params, bm = parse_sticker(bet_info.sticker)
                enp_id = key[0]

                if market in (BasketballMarkets.FULL_TIME_POINT_SPREAD, BasketballMarkets.FULL_TIME_MONEYLINE,
                              BasketballMarkets.FULL_TIME_TOTAL_POINTS):

                    n_win, n_bet = determine_basketball_outcome_from_api(market, params, enp_id)

                    report_bet_outcome_pnl(n_win, n_bet, bet_info)
                else:
                    logging.error("Cannot determine market pnl #Order %s\n" % bet_info)
                    continue

    def report_missing_fx_rate(self):
        for _, orders in self.orders_by_provider.iteritems():
            for bet_info in orders:
                if reported_fx_rate(bet_info.account_currency, bet_info.exchange_rate):
                    logging.error("#Order: \n%s\n", bet_info)


def reported_pnl(order_pnl, expected_pnl):
    if order_pnl >= 0 and expected_pnl >= 0 or order_pnl < 0 and expected_pnl < 0:
        if abs(order_pnl - expected_pnl) >= 10:
            logging.error("#PNL: Actual [%2f], Expected [%2f]" % (order_pnl, expected_pnl))
            return True
    else:
        logging.error("#PNL: Actual [%2f], Expected [%2f]" % (order_pnl, expected_pnl))
        return True

    return False


def report_bet_outcome_pnl(n_win, n_bet, bet_info):
    outcome, expected_pnl = determine_order_outcome_pnl(bet_info, n_win, n_bet)
    order_pnl = bet_info.gross

    if reported_pnl(order_pnl, expected_pnl):
        logging.error("#Order: \n%s\n", bet_info)


def reported_fx_rate(currency, fx_rate):
    if currency != "GBP" and fx_rate == 1:
        logging.error("#Currency: %s with fx rate %.2f" % (currency, fx_rate))
        return True
    else:
        return False
