import bisect
import numbers
from collections import defaultdict
import numpy as np
from datetime import timedelta, date

from sgmtradingcore.analytics.incremental_moving_averages import get_new_mean, get_new_var
from sgmtradingcore.core.ts_log import get_logger
from stratagemdataprocessing.bookmakers.common.date_utilities import get_date


class IncrementalMetrics(object):
    """
    Calculate metrics, as much as possible on every order update. Annualised numbers
    obviously need to use mean and volatily of daily returns but compound return and
    drawdown are calculated based on every order
    """

    def __init__(self, capital_by_day, trading_dates, num_trading_days):
        self._trading_dates = trading_dates
        self._num_trading_days = num_trading_days

        self._previous_date = None
        self._current_date = date.min
        self._current_date_pnl = None
        self._capital_by_day = capital_by_day  # date to capital

        self._total_pnl_by_day = {}  # date to total pnl up until that day, updated when the day ends
        self._current_total_pnl = 0

        self._mean_return_by_day = {}  # date to mean return up until that day, updated when the day ends
        self._variance_return_by_day = {}  # date to variance of return up until that day, updated when the day ends
        self._current_mean_return = None
        self._current_variance_return = None
        self._cumsum_return_by_day = {}  # date to cumsum of return up until that day, updated when the day ends
        self._current_cumsum_return = None

        self._return = 0
        self._volatility = 0
        self._volatility_annualised = 0
        self._sharpe_ratio = np.nan

        self._peak_return_plus_1 = float('-inf')
        self._max_drawdown = 0
        self._median_w = 0
        self._median_l = 0
        self._won_order = []
        self._lost_order = []

        self._unique_trade_ids = set()

        self._logger = get_logger(__name__)

    def update_return_structures(self, update_daily_structures):
        num_trading_days = self._num_trading_days

        first_day = self._current_date == date.min or self._previous_date == date.min

        if isinstance(self._capital_by_day, numbers.Number):
            capital = float(self._capital_by_day)
        else:
            capital = float(self._capital_by_day[self._current_date])
            if capital == 0.:
                day = self._current_date-timedelta(days=1)
                while day in self._capital_by_day and capital == 0.:
                    capital = self._capital_by_day[day]
                    day = day - timedelta(days=1)
        if capital <= 0.:
            raise ValueError("Capital {} on date {}".format(capital, self._current_date))

        current_day_return = self._current_date_pnl / capital

        # calculate new metrics, the calculations are much simpler on the first day
        if first_day:
            new_total_pnl = self._current_date_pnl

            new_mean_return = current_day_return

            new_cumsum_return = current_day_return
            new_return = new_cumsum_return

            new_variance_return = 0
            new_volatility = 0

            new_volatility_annualised = 0
            new_sharpe_ratio = np.nan
        else:
            new_total_pnl = self._total_pnl_by_day[self._previous_date] + self._current_date_pnl

            prev_day_mean_return = self._mean_return_by_day[self._previous_date]
            num_days = len(self._mean_return_by_day) + 1
            new_mean_return = get_new_mean(prev_day_mean_return, current_day_return, num_days)

            new_cumsum_return = self._cumsum_return_by_day[self._previous_date] + current_day_return
            new_return = new_cumsum_return

            prev_day_variance_return = self._variance_return_by_day[self._previous_date]
            new_variance_return = get_new_var(
                prev_day_variance_return, prev_day_mean_return, current_day_return, num_days)
            new_volatility = np.sqrt(new_variance_return) * np.sqrt(num_days)

            new_volatility_annualised = np.sqrt(new_variance_return) * np.sqrt(num_trading_days)
            if new_volatility == 0.:
                new_sharpe_ratio = np.nan
            else:
                new_sharpe_ratio = new_return / new_volatility

        # calculate drawdown - on every order update
        new_return_plus_1 = new_cumsum_return

        if new_return_plus_1 > self._peak_return_plus_1:
            self._peak_return_plus_1 = new_return_plus_1

        # note here drawdown is positive but we store it as negative
        # we no longer do a % change just the absolute difference of the compound returns (obviously the +1s cancel out)
        drawdown = self._peak_return_plus_1 - new_return_plus_1

        if drawdown > abs(self._max_drawdown):
            self._max_drawdown = -drawdown

        # update metrics
        self._current_total_pnl = new_total_pnl
        self._current_mean_return = new_mean_return
        self._current_cumsum_return = new_cumsum_return

        self._current_variance_return = new_variance_return

        self._return = new_return
        self._volatility = new_volatility
        self._volatility_annualised = new_volatility_annualised
        self._sharpe_ratio = new_sharpe_ratio

        if update_daily_structures:
            self._total_pnl_by_day[self._current_date] = new_total_pnl
            self._mean_return_by_day[self._current_date] = new_mean_return
            self._cumsum_return_by_day[self._current_date] = new_cumsum_return
            self._variance_return_by_day[self._current_date] = new_variance_return

    def update_metrics(self, order):
        order_date = order.settled_dt.date()

        if order_date < self._current_date:
            self._logger.warn(None, 'Ignoring order %s with old settled date' % (str(order)))
            return

        if order_date > self._current_date:
            # day change
            if self._current_date != date.min:
                self.update_return_structures(True)

            # handle any skipped trading dates
            if self._current_date != date.min:
                missing_trading_dates = get_missing_trading_dates(self._current_date, order_date, self._trading_dates)
                for missing_trading_date in missing_trading_dates:
                    self._previous_date = self._current_date
                    self._current_date = missing_trading_date
                    self._current_date_pnl = 0
                    self.update_return_structures(True)

            self._previous_date = self._current_date
            self._current_date = order_date
            self._current_date_pnl = 0

        self._current_date_pnl += order.pnl
        self.update_return_structures(False)

    def update_rina(self, order):

        if isinstance(self._capital_by_day, numbers.Number):
            capital = self._capital_by_day
        else:
            capital = self._capital_by_day[self._current_date]
            if capital <= 0.0:
                capital = self._capital_by_day[order.placed_dt.date()]
        capital = float(capital)
        if capital <= 0.0:
            raise ValueError("Placed bet with allocated capital {} on date {}".format(capital, order.placed_dt.date()))

        pnl = float(order.pnl)
        if pnl > 0:
            bisect.insort_left(self._won_order, pnl / capital)
        elif pnl < 0:
            bisect.insort_left(self._lost_order, abs(pnl) / capital)

    def update_rina_metric(self):
        len_w = len(self._won_order)
        len_l = len(self._lost_order)

        if len_w % 2 == 0:
            first_index = len_w / 2 - 1
            second_index = first_index + 1
            if len_w == 0:
                median_w = 0
            else:
                median_w = (self._won_order[first_index] + self._won_order[second_index]) / 2
        else:
            index = (len_w - 1) / 2
            median_w = self._won_order[index]

        if len_l % 2 == 0:
            first_index = len_l / 2 - 1
            second_index = first_index + 1
            if len_l == 0:
                median_l = 0
            else:
                median_l = (self._lost_order[first_index] + self._lost_order[second_index]) / 2
        else:
            index = (len_l - 1) / 2
            median_l = self._lost_order[index]

        self._median_w = median_w
        self._median_l = median_l

    def flat_capital_metrics_incremental(self, orders):
        process_orders = orders

        for order in process_orders:
            self._unique_trade_ids.add(order.trade_id)
            self.update_metrics(order)  # the return metrics
            self.update_rina(order)

        self.update_rina_metric()

        metrics = {
            'total_pnl': self._current_total_pnl,
            'maximum_drawdown': self._max_drawdown,
            'cum_return': self._return or 0,
            'volatility': self._volatility,
            'volatility_annualised': self._volatility_annualised,
            'sharpe_ratio': self._sharpe_ratio,
            'n_trades': len(self._unique_trade_ids),
            'median_l': self._median_l,
            'median_w': self._median_w
        }

        return metrics


def get_missing_trading_dates(start, end, trading_dates):
    if start > end:
        return []

    missing_traded_days = []
    missing = ((end - start) - timedelta(days=1)).days

    current = start
    for i in range(missing):
        current += timedelta(days=1)
        if current > end:
            break
        if current in trading_dates:
            missing_traded_days.append(current)
    return missing_traded_days
