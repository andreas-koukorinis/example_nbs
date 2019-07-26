from collections import defaultdict
from datetime import datetime, timedelta

import logging
import math
import pytz
from dateutil import parser

from sgmtradingcore.analytics.execution.benchmarks import exchange_vwap
from sgmtradingcore.analytics.results.common import determine_order_outcome_pnl
from sgmtradingcore.analytics.results.tennis import determine_tennis_outcome_from_api
from sgmtradingcore.execution.monitoring import json_to_instruction
from sgmtradingcore.strategies.strategy_base import StrategyStyle
from stratagemdataprocessing import data_api
from stratagemdataprocessing.bookmakers.common.odds.cache import HistoricalOddsCache
from stratagemdataprocessing.enums.odds import Sports, Bookmakers
from stratagemdataprocessing.parsing.common.stickers import parse_sticker

logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - ''%(levelname)s - %(message)s')

start_dt = datetime(2017, 1, 1, 0, tzinfo=pytz.UTC)
end_dt = datetime(2017, 3, 7, 23, 59, tzinfo=pytz.UTC)

trading_user_id = '54da2b5fd47e6bff0dade9b4'
# trading_user_id = '562f5bef497aee1c22000001'

strategy_ids = ['tennis_sip_ATP', 'tennis_sip_WTA', 'tennis_sip_v2_ATP', 'tennis_sip_v2_WTA']

pnl_delta = defaultdict(float)
return_delta = defaultdict(float)

cache = HistoricalOddsCache()

for strategy_id in strategy_ids:
    capital_by_date = data_api.get_capital_timeseries(
        trading_user_id, Sports.TENNIS, StrategyStyle.to_str(StrategyStyle.INPLAY),
        start_dt.date(), end_dt.date(), strategy_id)

    instructions = data_api.get_closed_instructions(trading_user_id, strategy_id)

    filtered_instructions = []

    for instr_json in instructions:
        if start_dt <= parser.parse(instr_json['placed_time']) <= end_dt:
            filtered_instructions.append(instr_json)

    for instr_json in filtered_instructions:
        sticker = instr_json['sticker']

        if instr_json['size_matched'] < instr_json['size']:
            remaining = instr_json['size'] - instr_json['size_matched']

            sport, market_scope, market, params, _ = parse_sticker(sticker)
            outcome, n_bet = determine_tennis_outcome_from_api(market, params, market_scope[1])

            placed_dt = parser.parse(instr_json['placed_time'])

            vwap = exchange_vwap(
                sticker, Bookmakers.BETFAIR, placed_dt, placed_dt + timedelta(seconds=50),
                odds_cache=cache, return_nan=True)

            instr = json_to_instruction(instr_json)
            instr.size = remaining
            instr.matched_amount = 0
            instr.matched_odds = 0 if math.isnan(vwap) else vwap
            instr.status = -1
            outcome, pnl = determine_order_outcome_pnl(instr, outcome, n_bet, default_if_unknown=True)

            if pnl > 0:
                pnl *= 0.97

            pnl_delta[strategy_id] += pnl
            return_delta[strategy_id] += pnl / capital_by_date[placed_dt.date()]
            # return_delta[]

print return_delta
