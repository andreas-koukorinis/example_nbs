from argparse import ArgumentTypeError

from bson.objectid import ObjectId
from cassandra.cluster import defaultdict
import copy
import math
import pytz
import logging
import argparse

from sgmtradingcore.core.trading_types import OrderSide
from datetime import datetime, time, timedelta

from sgmtradingcore.execution.monitoring import json_to_bet_info
from stratagemdataprocessing.crypto.enums import Asset, get_parts, CryptoExchange, CryptoMarkets, get_assets
from stratagemdataprocessing.dbutils.mongo import MongoPersister


def valid_string_list(s):
    try:
        result = [x.strip() for x in s.split(',')]
        return result
    except BaseException:
        msg = "Not a valid strategy lis: '{0}'.".format(s)
        raise ArgumentTypeError(msg)


def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise ArgumentTypeError(msg)

TZ = pytz.timezone('Europe/London')
CLOSING_TIME = time(17, 0, 0)


def _sign(x):
    return math.copysign(1, x)


class PnlContainer(object):

    def __init__(self, size, currency):
        self.size = size
        self.currency = currency

    def __add__(self, other):
        if other.currency != self.currency:
            msg = 'Cannot add pnl containers of difference currencies %s and %s' % (
                self.currency, other.currency)
            raise ValueError(msg)

        new_pnl = PnlContainer(self.size + other.size, self.currency)
        return new_pnl

    def __iadd__(self, other):
        if other.currency != self.currency:
            msg = 'Cannot add pnl containers of difference currencies %s and %s' % (
                self.currency, other.currency)
            raise ValueError(msg)

        self.size = self.size + other.size


class PnlCalculator(object):
    """
    A Pnl calculator knows how to calculate for a given contract
    and entry/exit prices and position size
    """

    def calculate_pnl(self, entry_price, exit_price, exit_size):
        raise NotImplementedError('Please implement this method')


class DefaultPnlCalculator(object):
    """
    Default calculator calculates value different in the quote currency
    """
    def __init__(self, quote_currency):
        self._quote_currency = quote_currency

    def calculate_pnl(self, entry_price, exit_price, exit_size):
        realized_pnl = exit_size * (exit_price - entry_price)
        return PnlContainer(realized_pnl, self._quote_currency)


class XBTPnlCalculator(object):

    def calculate_pnl(self, entry_price, exit_price, exit_size):
        realized_pnl = exit_size * (1./entry_price - 1./exit_price)
        return PnlContainer(realized_pnl, Asset.BTC)

class ETHPnlCalculator(object):

    def calculate_pnl(self, entry_price, exit_price, exit_size):
        realized_pnl = exit_size * (exit_price - entry_price) * 10**-6
        return PnlContainer(realized_pnl, Asset.BTC)

class LTCPnlCalculator(object):

    def calculate_pnl(self, entry_price, exit_price, exit_size):
        realized_pnl = exit_size * (exit_price - entry_price)
        return PnlContainer(realized_pnl, Asset.BTC)

class ADAPnlCalculator(object):

    def calculate_pnl(self, entry_price, exit_price, exit_size):
        realized_pnl = exit_size * (exit_price - entry_price)
        return PnlContainer(realized_pnl, Asset.BTC)

class XRPPnlCalculator(object):

    def calculate_pnl(self, entry_price, exit_price, exit_size):
        realized_pnl = exit_size * (exit_price - entry_price)
        return PnlContainer(realized_pnl, Asset.BTC)

def create_pnl_calculator(ticker):
    cross, market, exchange = get_parts(ticker)
    asset, quote_currency = get_assets(ticker)

    if market == CryptoMarkets.SPOT:
        return DefaultPnlCalculator(quote_currency)
    elif asset in (Asset.BTCUSD_PERP, Asset.BTCUSD_FUTU18, Asset.BTCUSD_FUTZ18, Asset.BTCUSD_FUTH19):
        return XBTPnlCalculator()
    elif asset in (Asset.ETHUSD_PERP):
        return ETHPnlCalculator()
    elif asset in (Asset.LTCBTC_FUTU18, Asset.LTCBTC_FUTZ18):
        return LTCPnlCalculator()
    elif asset in (Asset.XRPBTC_FUTU18, Asset.XRPBTC_FUTZ18):
        return XRPPnlCalculator()
    elif asset in (Asset.ADABTC_FUTU18, Asset.ADABTC_FUTZ18):
        return ADAPnlCalculator()
    else:
        raise ValueError('Cannot create pnl calculator for %s', ticker)


class PositionContainer(object):
    """
    Position container keeps track of positions, accumulates realized PN

    Given a new fill and the current state(position + avg entry price),
    calculate any realized pnl and the new state

    - If the position increases (long or short) then just recalculate new
    average entry price of increased position
    - If the position decreases, note realised PNL (using last average entry & exit price)
      - If the position did not flip, average entry stays the same
      - If the position did flip calculate average entry using new trade price

    Detailed explanation here:
    https://www.tradingtechnologies.com/help/fix-adapter-reference/pl-calculation-algorithm/understanding-pl-calculations/
    """

    def __init__(self, size, entry_price, pnl_calculator):
        self.size = size
        self.entry_price = entry_price
        self._pnl_calculator = pnl_calculator
        self._realized_pnl = self._pnl_calculator.calculate_pnl(100, 100, 1)

    def update_position(self, trade):
        trade_size = trade.size
        trade_price = trade.price

        if _sign(self.size) != _sign(trade_size):
            size_closed = min(abs(self.size), abs(trade_size)) * _sign(trade_size)
            size_opened = trade_size - size_closed
        else:
            size_closed = 0
            size_opened = trade_size

        # TODO formalize EPSILON?
        new_position_size = self.size + trade_size
        if abs(new_position_size) < 1e-8:
            new_position_size = 0

        if self.size == 0:
            # first trade after being flat
            new_entry_price = trade_price
            realized_pnl = self._pnl_calculator.calculate_pnl(100, 100, 1)
        else:
            old_cost = self.entry_price * self.size
            new_cost = (
                old_cost + size_opened * trade_price +
                size_closed * old_cost / self.size
            )
            realized_pnl = self._pnl_calculator.calculate_pnl(self.entry_price, trade_price, -size_closed)
            new_entry_price = new_cost / new_position_size if new_position_size != 0 else None

        self.size = new_position_size
        self.entry_price = new_entry_price
        self._realized_pnl = self._realized_pnl + realized_pnl

    def realized_pnl(self):
        return self._realized_pnl

    def unrealized_pnl(self, current_price):
        unrealized_pnl = self._pnl_calculator.calculate_pnl(self.entry_price, current_price, self.size)
        return unrealized_pnl

    def snapshot(self):
        return PositionSnapshot(
            self.size, self.entry_price, self._realized_pnl.size, self._realized_pnl.currency)


class PositionSnapshot(object):
    """
    A frozen position container to save to the database and allows
    us to render the rows for the previous days
    """

    def __init__(self, size, entry_price, realized_pnl, pnl_currency):
        self.size = size
        self.entry_price = entry_price
        self._realized_pnl = realized_pnl
        self._pnl_currency = pnl_currency

    def update_position(self, trade):
        raise ValueError('Cannot update position of a position snapshot')

    def realized_pnl(self):
        return PnlContainer(self._realized_pnl, self._pnl_currency)

    def unrealized_pnl(self, current_price):
        return PnlContainer(0, 'USD')

    def to_dict(self):
        result = {
            'size': self.size,
            'entry_price': self.entry_price,
            'realized_pnl': self._realized_pnl,
            'pnl_currency': self._pnl_currency,
        }

        return result

    @classmethod
    def from_dict(cls, dict_):
        result = cls(
            dict_['size'], dict_['entry_price'], dict_['realized_pnl'], dict_['pnl_currency'],
        )
        return result


def filter_trades(period_start, period_end, orders_image, known_ids):
    """
    Extract trades from orders and filter for within the given period and list of ids
    :param period_start:(datetime) Start period generally 5pm of the current date
    :param period_end: (datetime) End period generally 5pm of the next day
    :param orders_image: (dict) Map of order id to Order
    :param known_ids: (set) Set of trade ids we have already processes
    """
    trades = defaultdict(list)

    # filter trades that happened in the relevant period
    for order_id, order in orders_image.iteritems():
        ticker = order.sticker
        if len(ticker) == 0:
            return dict()
            # raise ValueError('Cannot snap inventory with missing tickers')

        fills = order.matched_bets
        for fill in fills:
            if fill.trade_id in known_ids:
                continue

            fill_dt = fill.dt
            if period_start <= fill_dt < period_end:
                fill_copy = copy.copy(fill)
                if order.side == OrderSide.SELL:
                    fill_copy.size = - fill.size
                trades[ticker].append(fill_copy)

    return dict(trades)


# a snapshot is just a map from ticker to a SinglePositionSnapshot
def snap_inventory(prev_snapshot, trades):

    new_snapshot = copy.copy(prev_snapshot)

    # calculate PNL and EOD snapshot using the previous snapshot and trades
    for ticker, trades in trades.iteritems():
        ticker_snapshot = prev_snapshot.get(ticker, PositionSnapshot(0, None, 0, ''))
        pnl_calculator = create_pnl_calculator(ticker)
        position_container = PositionContainer(
            ticker_snapshot.size, ticker_snapshot.entry_price, pnl_calculator)

        for trade in trades:
            position_container.update_position(trade)

        new_snapshot[ticker] = position_container.snapshot()

    return new_snapshot


# a snapshot is a map from strategy to a map from ticker to a SinglePositionSnapshot
def load_snapshot(crypto_trading_conn, trading_user_id, date_, strategy=None):
    result = defaultdict(dict)

    query = {
        'trading_user_id': trading_user_id,
        'date': date_.strftime('%Y-%m-%d'),
    }
    if strategy is not None:
        query['strategy'] = strategy

    docs = crypto_trading_conn['position_snapshots'].find(query)

    for doc in docs:
        doc_strategy = doc['strategy']
        pos = PositionSnapshot(doc['size'], doc['entry_price'], doc['realized_pnl'], doc['pnl_currency'])
        result[doc_strategy][doc['ticker']] = pos

    return dict(result)


# a position snapshot is a map from strategy to a map from ticker to a SinglePositionSnapshot
# a pnl snapshot is a map from strategy to a map from ticker to the days pnl
def save_snapshot(crypto_trading_conn, trading_user_id, date_, position_snapshot):
    docs = []
    for strategy, strategy_snapshot in position_snapshot.iteritems():
        for ticker, snapshot in strategy_snapshot.iteritems():
            doc = {
                'trading_user_id': trading_user_id,
                'strategy': strategy,
                'date': date_.strftime('%Y-%m-%d'),
                'ticker': ticker,
            }
            doc.update(snapshot.to_dict())
            docs.append(doc)

    if len(docs):
        crypto_trading_conn['position_snapshots'].remove({'date': doc['date']})
        crypto_trading_conn['position_snapshots'].insert_many(docs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - ''%(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Daily inventory manager')
    parser.add_argument(
        '--trading_user_id', required=True, help='Trading user ID')
    parser.add_argument(
        '--strategies', required=True, help='Strategy1,Strategy2', type=valid_string_list)
    parser.add_argument(
        '--startdate', help='First date to close format YYYY-MM-DD ', required=True, type=valid_date)
    parser.add_argument(
        '--enddate', help='Last date to close - format YYYY-MM-DD ', required=True, type=valid_date)

    args = parser.parse_args()
    trading_user_id = ObjectId(args.trading_user_id)
    strategies = args.strategies

    crypto_trading_conn = MongoPersister.init_from_config('crypto_trading', auto_connect=True).db

    # look for trades in orders from up to 7 days ago
    load_start_date = args.startdate - timedelta(days=7)
    load_end_date = datetime.combine(args.enddate, CLOSING_TIME)


    all_orders = list(
        crypto_trading_conn['orders'].find({
            'trading_user_id': ObjectId(trading_user_id),
            'placed_time': {'$gte': load_start_date, '$lt': load_end_date},
        })
    )

    orders_by_strategy = defaultdict(dict)
    for order in all_orders:
        if len(order.get('sticker', '')) == 0:
            continue
        strategy = order.get('strategy', '')
        order_id = str(order['id'])
        orders_by_strategy[strategy][order_id] = json_to_bet_info(order, crypto=True, parse_dts=False)

    current_date = args.startdate
    previous_date = current_date - timedelta(days=1)
    current_date_snapshot = load_snapshot(crypto_trading_conn, trading_user_id, previous_date)

    while current_date <= args.enddate:
        logging.info('Snapshotting for %s', current_date)
        period_start = TZ.localize(datetime.combine(current_date - timedelta(days=1), CLOSING_TIME))
        period_end = TZ.localize(datetime.combine(current_date, CLOSING_TIME))

        for strategy in strategies:
            orders = orders_by_strategy.get(strategy, [])
            strategy_snapshot = current_date_snapshot.get(strategy, {})
            trades = filter_trades(period_start, period_end, orders, set())
            current_date_snapshot[strategy] = snap_inventory(
                strategy_snapshot, trades)

        save_snapshot(
            crypto_trading_conn, trading_user_id, current_date, current_date_snapshot)

        current_date += timedelta(days=1)


