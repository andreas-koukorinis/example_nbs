from collections import defaultdict
from copy import deepcopy
import inspect

from sgmtradingcore.analytics.incremental_moving_averages import get_new_mean
from sgmtradingcore.analytics.results.football import football_pnl_grid
from sgmtradingcore.core.trading_types import InstructionStatus
from sgmtradingcore.util.keyed_sets import KeyedSet
import numpy as np
from stratagemdataprocessing.enums.markets import Markets
from stratagemdataprocessing.enums.odds import Sports
from stratagemdataprocessing.parsing.common.stickers import parse_sticker, extract_sport
import warnings


def ks():
    return KeyedSet([], key=lambda x: x.id)


def empty_pnl_grid(grid_size=8):
    default_pnl_grid = np.full((grid_size, grid_size), 0., np.float64)
    return default_pnl_grid


class IncrementalMaxLoss(object):
    """
    Calculate max loss per trade incrementally. Only active orders count towards max loss
    so it makes sense to combine the numbers with incremental pnl to assess true risk/position.

    Also calculate time in market per trade, either using hints places in the orders by the strategy
    or by waiting for the pnl grid to flatten.
    """

    def __init__(self):
        self._current_pnl_grid_by_trade_by_event = defaultdict(
            lambda: defaultdict(empty_pnl_grid))  # {event_id: {trade_id: max loss grid}} - matched amount
        self._active_pnl_grid_by_trade_by_event = defaultdict(
            lambda: defaultdict(empty_pnl_grid))  # {event_id: {trade_id: max loss grid}} - whole amount
        self._max_loss_by_trade_by_event = defaultdict(dict)  # {event_id: {trade_id: max loss}}
        self._max_loss = None

        self._current_pnl_grid_by_instruction_id = dict()  # for the matched amount
        self._active_pnl_grid_by_instruction_id = dict()  # for the whole matchable amount

        self._instructions_by_trade_by_event = defaultdict(lambda: defaultdict(ks))  # {trade_id: {event_id: {instructions} } }

        self._trade_open_dt = {}
        self._trade_close_dt = {}

        self._time_in_market_by_trade = {}
        self._mean_time_in_market = 0

    def _update_max_loss(self, updated_instructions, _updated_orders):
        closed_trades = set()

        for instruction in updated_instructions:
            sport, (_, event_id), market, params, bm = parse_sticker(instruction.sticker)

            trade_id = instruction.trade_id

            if sport == Sports.FOOTBALL:
                # if not already present, initialise pnl grids
                # (accounting for the possibility of a corner market, which affects the size of the grids)
                # note: this effectively replaces the default grid of the defaultdict
                max_evts = _get_football_max_evts(instruction)

                if event_id not in self._current_pnl_grid_by_trade_by_event:
                    self._current_pnl_grid_by_trade_by_event[event_id] = {}
                if trade_id not in self._current_pnl_grid_by_trade_by_event[event_id]:
                    self._current_pnl_grid_by_trade_by_event[event_id][trade_id] = empty_pnl_grid(
                        grid_size=max_evts + 1)

                if event_id not in self._active_pnl_grid_by_trade_by_event:
                    self._active_pnl_grid_by_trade_by_event[event_id] = {}
                if trade_id not in self._active_pnl_grid_by_trade_by_event[event_id]:
                    self._active_pnl_grid_by_trade_by_event[event_id][trade_id] = empty_pnl_grid(
                        grid_size=max_evts + 1)
            else:
                err_msg = ('WARN: risk not implemented for sport {}'
                           .format(instruction.sticker))
                warnings.warn(err_msg)

            if instruction in self._instructions_by_trade_by_event[event_id][trade_id]:
                self._current_pnl_grid_by_trade_by_event[event_id][trade_id] -= self._current_pnl_grid_by_instruction_id[instruction.id]
                self._active_pnl_grid_by_trade_by_event[event_id][trade_id] -= self._active_pnl_grid_by_instruction_id[instruction.id]

            self._instructions_by_trade_by_event[event_id][trade_id].add(instruction)

            # create an instruction for the total matchable amount
            instruction_copy = deepcopy(instruction)
            if instruction.status in (InstructionStatus.IDLE, InstructionStatus.PROCESSING, InstructionStatus.WATCHING):
                instruction_copy.matched_amount = instruction.size
            else:
                instruction_copy.matched_amount = instruction.matched_amount

            sport = extract_sport(instruction.sticker)

            if sport == Sports.FOOTBALL:
                max_evts = _get_football_max_evts(instruction)
                self._current_pnl_grid_by_instruction_id[instruction.id] = football_pnl_grid(
                    [instruction], max_evts=max_evts, reshape=True)
                self._active_pnl_grid_by_instruction_id[instruction.id] = football_pnl_grid(
                    [instruction_copy], max_evts=max_evts, reshape=True)
            else:
                # TODO more sports
                self._current_pnl_grid_by_instruction_id[instruction.id] = empty_pnl_grid()
                self._active_pnl_grid_by_instruction_id[instruction.id] = empty_pnl_grid()

            self._current_pnl_grid_by_trade_by_event[event_id][trade_id] += self._current_pnl_grid_by_instruction_id[instruction.id]
            self._active_pnl_grid_by_trade_by_event[event_id][trade_id] += self._active_pnl_grid_by_instruction_id[instruction.id]

            # check if a trade has opened, the open dt will be that provided by the strategy,
            # this is for RINA style calculations - this is now the only way
            # to calculate time in market, there is no fancy attempts to automatically calculate it
            if trade_id not in self._trade_open_dt and 'trade_open_dt' in instruction.details:
                trade_open_dt = instruction.details['trade_open_dt']
                self._trade_open_dt[trade_id] = trade_open_dt

            # check if a trade has been labelled closed by the strategy
            if trade_id in self._trade_open_dt and trade_id not in self._trade_close_dt and 'trade_close_dt' in instruction.details:
                trade_closed_dt = instruction.details['trade_close_dt']
                self._trade_close_dt[trade_id] = trade_closed_dt
                closed_trades.add(trade_id)

        # TODO consider reducing max loss as orders settle?
        max_loss = 0

        # calculate the max loss from the active pnl grids, ie the total matchable amounts
        for event_id, pnl_grid_by_trade in self._active_pnl_grid_by_trade_by_event.items():
            max_loss_for_event = 0

            for trade_id, pnl_grid_for_trade in pnl_grid_by_trade.items():
                min_pnl_for_trade = np.min(pnl_grid_for_trade)
                max_loss_for_trade = min(0, min_pnl_for_trade)
                self._max_loss_by_trade_by_event[event_id][trade_id] = max_loss_for_trade
                max_loss_for_event += max_loss_for_trade

            max_loss += max_loss_for_event

        self._max_loss = max_loss

        for closed_trade_id in closed_trades:
            if closed_trade_id not in self._time_in_market_by_trade:
                time_in_market_sec = (self._trade_close_dt[closed_trade_id] - self._trade_open_dt[closed_trade_id]).total_seconds()
                time_in_market = min(1., time_in_market_sec / (111. * 60.))
                self._time_in_market_by_trade[closed_trade_id] = time_in_market

                new_mean_time_in_market = get_new_mean(
                    self._mean_time_in_market, time_in_market, len(self._time_in_market_by_trade))
                self._mean_time_in_market = new_mean_time_in_market

    def max_loss_incremental(self, updated_instructions, updated_orders):
        self._update_max_loss(updated_instructions, updated_orders)

        return {
            'max_loss': self._max_loss,
            'max_loss_by_trade_by_event': self._max_loss_by_trade_by_event,
            'time_in_market': self._mean_time_in_market,
            'current_pnl_grid_by_trade_by_event': self._current_pnl_grid_by_trade_by_event,
            'active_pnl_grid_by_trade_by_event': self._active_pnl_grid_by_trade_by_event,
        }


def _get_football_max_evts(instruction):
    _, _, market, _, _ = parse_sticker(instruction.sticker)
    return (25 if Markets.is_corner_mkt(market)
            else _get_default_args(football_pnl_grid)['max_evts'])


def _get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(args[-len(defaults):], defaults))
