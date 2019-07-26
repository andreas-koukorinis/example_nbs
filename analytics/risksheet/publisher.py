import copy
import traceback
from datetime import datetime, timedelta, time
from threading import Thread, Condition, Event, Lock
import logging
import pytz
import argparse
from cassandra.io.libevreactor import LibevConnection

from itertools import chain
from bson.objectid import ObjectId

from sgmtradingcore.analytics.risksheet.inventory import (
    load_snapshot, filter_trades, PositionContainer, CLOSING_TIME, TZ, create_pnl_calculator, PnlContainer)
from sgmtradingcore.execution.monitoring import json_to_bet_info
from sgmtradingcore.providers.crypto.trades.trades_provider import json_to_crypto_trade
from stratagemdataprocessing.crypto.enums import Asset, generate_crypto_ticker, CryptoMarkets, CryptoExchange, get_parts, \
    get_assets
from stratagemdataprocessing.crypto.market.cassandra_data import get_cassandra_trades_ticks
from stratagemdataprocessing.dbutils.cassandradb import CassandraConnector
from stratagemdataprocessing.dbutils.mongo import MongoPersister
from stratagemdataprocessing.tablestream.protocol.transport import RabbitMQTransport
from stratagemdataprocessing.tablestream.publisher.publisher import TableStreamPublisher
from stratagemdataprocessing.tablestream.subscriber.subscriber import TableStreamSubscriber
from stratagemdataprocessing.util.thread_manager import DaemonThreadExceptionManager

NAN = 'NaN'

class RiskSheetPublisher(object):
    def __init__(self, trading_user_id, subscriber, publisher, crypto_trading_conn, cassandra_conn,
                 current_date, thread_manager=None):

        self._trading_user_id = trading_user_id
        self._subscriber = subscriber
        self._publisher = publisher
        self._crypto_trading_conn = crypto_trading_conn
        self._cassandra_conn = cassandra_conn
        self._thread_manager = thread_manager
        self._event = Event()
        self.ready = False  # flag for unit test to subscribe at the right time

        self._orders_table_name = 'trading_user_orders_%s' % self._trading_user_id
        self._risksheet_table_name = 'trading_user_risksheet_%s' % self._trading_user_id
        self._inputs_table_name = 'trading_user_risk_inputs_%s' % self._trading_user_id
        self._last_published_dt = datetime.min.replace(tzinfo=pytz.UTC)

        self._current_date = current_date
        self._positions_by_date = {}  # date_str -> strategy -> asset -> position
        self._historical_closes = {} # date_str -> asset -> price

        self._last_trade_tick = {}  # ticker to trade tick

        self._strategy_by_order_id = {}
        self._all_trade_ids = set()

        self._sigma = 165.0
        self._beta = 2.20

    # tablestream subscriber
    def heartbeat_status(self, table_name, up):
        if not up:
            msg = 'Received tablestream heartbeat status down for table %s' % table_name
            raise ValueError(msg)
        else:
            msg = 'Received tablestream heartbeat status up for table %s' % table_name
            logging.info(msg)

    def subscription_failed(self, table_name, err):
        logging.error('Received tablestream subscription error for table %s' % table_name)
        raise err

    def image_updated(self, table_name, image):
        if table_name == self._orders_table_name:
            self._handle_orders_update(image, True)
        elif table_name == self._inputs_table_name:
            self._handle_inputs_update(image, True)
        elif table_name.startswith('crypto_'):
            ticker = table_name.split('_')[1]
            self._handle_md_update(ticker, image, True)
        else:
            logging.warn('Ignoring update from table %s', table_name)

    def received_delta(self, table_name, changes):
        if table_name == self._orders_table_name:
            self._handle_orders_update(changes, False)
        elif table_name == self._inputs_table_name:
            self._handle_inputs_update(changes, True)
        elif table_name.startswith('crypto_'):
            ticker = table_name.split('_')[1]
            self._handle_md_update(ticker, changes, False)
        else:
            logging.warn('Ignoring update from table %s', table_name)

    def _handle_inputs_update(self, updated, reset):

        if 'sigma' in updated:
            self._sigma = updated['sigma']
        if 'beta' in updated:
            self._beta = updated['beta']

        logging.info('Risk inputs updated: sigma=%s beta=%s' % (self._sigma, self._beta))
        image = self._create_positions_image()
        self._publisher.set_image(self._risksheet_table_name, image)
        self._last_published_dt = datetime.now(pytz.UTC)

    def _handle_orders_update(self, updated, reset):

        if reset:
            logging.info('Resetting all positions')
            if self._current_date in self._positions_by_date:
                del self._positions_by_date[self._current_date]

            self._strategy_by_order_id = {}
            self._all_trade_ids = set()

            # copy previous day positions
            self._positions_by_date[self._current_date] = {}
            previous_day_positions = self._positions_by_date[self._current_date - timedelta(days=1)]
            for strategy, strategy_container in previous_day_positions.iteritems():
                for ticker, position_container in strategy_container.iteritems():
                    pnl_calculator = create_pnl_calculator(ticker)
                    if ticker not in self._last_trade_tick:
                        self._subscribe_to_trade_data(ticker)

                    if strategy not in self._positions_by_date[self._current_date]:
                        self._positions_by_date[self._current_date][strategy] = {}
                    self._positions_by_date[self._current_date][strategy][ticker] =\
                        PositionContainer(position_container.size, position_container.entry_price, pnl_calculator)

        for order_id, order_json in updated.iteritems():
            if order_id.startswith('error'):
                continue

            order = json_to_bet_info(order_json, crypto=True)

            if order.sticker == '' or order.sticker.startswith('undefined'):
                logging.warning('Ignoring order %s with empty sticker' % order.id)
                continue

            new_order = order.id not in self._strategy_by_order_id
            if not new_order:
                old_strategy = self._strategy_by_order_id[order.id]
                if old_strategy != order.strategy_id:
                    # retagging of order - reset the position
                    # TODO check if something got retagged in a previous date and fail
                    # simply unsubscribe and resubscribe to reset current day positions
                    self._subscriber.unsubscribe([self._orders_table_name])
                    self._subscriber.subscribe([self._orders_table_name], publisher='crypto_oms')
                    return

            self._strategy_by_order_id[order.id] = order.strategy_id
            ticker_exchange = order.sticker

            # check if we need to subscribe for market data for the ticker itself
            if ticker_exchange not in self._last_trade_tick:
                self._subscribe_to_trade_data(ticker_exchange)

            # and also for the spot pair
            cross, market, exchange = get_parts(ticker_exchange)
            spot_ticker_exchange = generate_crypto_ticker(cross, CryptoMarkets.SPOT, CryptoExchange.GDAX)
            if spot_ticker_exchange not in self._last_trade_tick:
                self._subscribe_to_trade_data(spot_ticker_exchange)

                # If asset is not BTC, subscribe to the BTC cross for conversion
                if not cross.startswith(Asset.BTC):
                    crypto_currency = cross[:3] # assumes all pairs have crypto at the start e.g. ETHUSD
                    if cross.endswith(Asset.USD):
                        btc_cross = crypto_currency + Asset.BTC # crypto-crypto crosses on gdax have BTC second
                        btc_cross_ticker = generate_crypto_ticker(btc_cross, CryptoMarkets.SPOT, CryptoExchange.GDAX)
                        if btc_cross_ticker not in self._last_trade_tick:
                            self._subscribe_to_trade_data(btc_cross_ticker)
                    else:
                        usd_cross = crypto_currency + Asset.USD
                        usd_cross_ticker = generate_crypto_ticker(usd_cross, CryptoMarkets.SPOT, CryptoExchange.GDAX)
                        if usd_cross_ticker not in self._last_trade_tick:
                            self._subscribe_to_trade_data(usd_cross_ticker)

            # get any new trades and update position
            current_period_end = TZ.localize(datetime.combine(self._current_date, CLOSING_TIME))
            current_period_start = current_period_end - timedelta(days=1)
            all_new_trades = filter_trades(
                current_period_start, current_period_end, {order.id: order}, self._all_trade_ids)

            new_trade_ids = {tr.trade_id for tr in chain(*all_new_trades.values())}
            self._all_trade_ids.update(new_trade_ids)

            current_day_positions = self._positions_by_date.setdefault(self._current_date, {})
            strategy_positions = current_day_positions.setdefault(order.strategy_id, {})

            if len(all_new_trades):
                if ticker_exchange not in strategy_positions:
                    try:
                        logging.info('Creating new position container for ticker %s' % ticker_exchange)
                        pnl_calculator = create_pnl_calculator(ticker_exchange)

                        previous_day_positions = self._positions_by_date.get(
                            self._current_date - timedelta(days=1), {})
                        previous_day_strategy_positions = previous_day_positions.get(
                            order.strategy_id, {})
                        previous_day_position_container = previous_day_strategy_positions.get(
                            ticker_exchange, PositionContainer(0, None, pnl_calculator))

                        # TODO replace previous date container with a snapshot

                        position_container = PositionContainer(
                            previous_day_position_container.size, previous_day_position_container.entry_price, pnl_calculator)
                        strategy_positions[ticker_exchange] = position_container
                    except:
                        logging.warn('Cannot calculate pnl for ticker %s' % ticker_exchange)
                        continue

                position_container = strategy_positions[ticker_exchange]
                for trades in all_new_trades.itervalues():
                    logging.info('Processing %d new trades for ticker %s' % (len(trades), ticker_exchange))
                    for trade in trades:
                        position_container.update_position(trade)

        # TODO do something more efficient later
        image = self._create_positions_image()
        self._publisher.set_image(self._risksheet_table_name, image)

    def _subscribe_to_trade_data(self, ticker):
        if ticker not in self._last_trade_tick:
            # Don't resubscribe to this ticker
            self._last_trade_tick[ticker] = None
            logging.info('Subscribing for trade data for ticker %s', ticker)
            md_table = 'crypto_%s_TRADE' % ticker
            self._subscriber.subscribe([md_table], publisher='ts_endpoint')

    def _handle_md_update(self, ticker, updated, initial):
        if len(updated):
            tick = json_to_crypto_trade(updated['data'])
            self._last_trade_tick[ticker] = tick

            now_dt = datetime.now(pytz.UTC)
            elapsed = now_dt - self._last_published_dt

            if elapsed > timedelta(seconds=5) or initial:
                image = self._create_positions_image()
                self._publisher.set_image(self._risksheet_table_name, image)
                self._last_published_dt = now_dt

    def calculate_exposure(self, ticker, position_container):
        exposure_usd = NAN
        cross, market, _ = get_parts(ticker)
        asset = cross[:3]
        quote_currency = cross[3:]
        if asset == Asset.USD:
            exposure_usd = position_container.size
        elif quote_currency == Asset.USD:
            if market == CryptoMarkets.SPOT:
                exposure_usd = self.convert_pnl(PnlContainer(position_container.size, asset), Asset.USD)
            elif market == CryptoMarkets.PERPETUAL or market == CryptoMarkets.FUTURE:
                exposure_usd = position_container.size
        elif quote_currency == Asset.BTC:
            if position_container.entry_price is not None:
                nav_btc = position_container.size * position_container.entry_price
                exposure_usd = self.convert_pnl(PnlContainer(nav_btc, Asset.BTC), Asset.USD)

        return exposure_usd

    def convert_pnl(self, pnl_container, target_currency):
        if pnl_container.currency == target_currency:
            return pnl_container.size
        else:
            # check if we have a last traded price for spot
            pair = '%s%s' % (pnl_container.currency, target_currency)
            try:
                spot_ticker = generate_crypto_ticker(pair, CryptoMarkets.SPOT, CryptoExchange.GDAX)
                if spot_ticker in self._last_trade_tick:
                    return pnl_container.size * self._last_trade_tick[spot_ticker].price
            except:
                pass

            reverse_pair = '%s%s' % (target_currency, pnl_container.currency)
            try:
                reverse_spot_ticker = generate_crypto_ticker(reverse_pair, CryptoMarkets.SPOT, CryptoExchange.GDAX)
                if reverse_spot_ticker in self._last_trade_tick:
                    return pnl_container.size / self._last_trade_tick[reverse_spot_ticker].price
            except:
                pass

        return NAN

    def _create_positions_image(self):
        image = {}
        for date_, date_container in self._positions_by_date.iteritems():
            for strategy, strategy_container in date_container.iteritems():
                for ticker, position_container in strategy_container.iteritems():

                    key = '%s-%s-%s' % (strategy, date_.strftime('%Y-%m-%d'), ticker)

                    realized_pnl = position_container.realized_pnl()
                    realized_pnl_usd = self.convert_pnl(realized_pnl, Asset.USD)
                    realized_pnl_btc = self.convert_pnl(realized_pnl, Asset.BTC)
                    exposure_usd = self.calculate_exposure(ticker, position_container)
                    unrealized_pnl_usd = NAN
                    unrealized_pnl_btc = NAN
                    cod_pnl_usd = NAN
                    cod_pnl_btc = NAN
                    pnl_total_usd = NAN
                    pnl_total_btc = NAN
                    live_price = NAN
                    sigma_price = NAN
                    sigma_pnl_usd = NAN

                    if ticker in self._last_trade_tick and self._last_trade_tick[ticker] is not None\
                            and position_container.entry_price is not None:

                        last_traded_price = self._last_trade_tick[ticker].price
                        live_price = last_traded_price
                        unrealized_pnl = position_container.unrealized_pnl(last_traded_price)
                        unrealized_pnl_usd = self.convert_pnl(unrealized_pnl, Asset.USD)
                        if realized_pnl_usd != NAN and unrealized_pnl_usd != NAN:
                            pnl_total_usd = realized_pnl_usd + unrealized_pnl_usd

                        unrealized_pnl_btc = self.convert_pnl(unrealized_pnl, Asset.BTC)
                        if realized_pnl_btc != NAN and unrealized_pnl_btc != NAN:
                            pnl_total_btc = realized_pnl_btc + unrealized_pnl_btc

                        # Calculate pnl change for a 1 sigma move
                        if date_ == self._current_date:
                            if ticker.startswith(Asset.BTC):
                                sigma_price = last_traded_price + self._sigma
                            else:
                                btc_spot_ticker = generate_crypto_ticker(Asset.BTC+Asset.USD, CryptoMarkets.SPOT, CryptoExchange.GDAX)
                                if btc_spot_ticker in self._last_trade_tick and self._last_trade_tick[btc_spot_ticker] is not None:
                                    last_traded_btc = self._last_trade_tick[btc_spot_ticker].price
                                    sigma_price = self._sigma / last_traded_btc * last_traded_price * self._beta + last_traded_price

                            if sigma_price != NAN and unrealized_pnl_usd != NAN:
                                sigma_pnl = position_container.unrealized_pnl(sigma_price)
                                sigma_pnl_usd = self.convert_pnl(sigma_pnl, Asset.USD) - unrealized_pnl_usd

                        # Calculate change on day
                        yesterday = date_ + timedelta(days=-1)
                        if yesterday in self._historical_closes and ticker in self._historical_closes[yesterday]\
                                and position_container.entry_price is not None:

                            close_tick = self._historical_closes[yesterday][ticker]
                            close_pnl = position_container.unrealized_pnl(close_tick.price)

                            if pnl_total_usd != NAN:
                                close_pnl_usd = self.convert_pnl(close_pnl, Asset.USD)
                                if close_pnl != NAN:
                                    cod_pnl_usd = pnl_total_usd - close_pnl_usd

                            if pnl_total_btc != NAN:
                                close_pnl_btc = self.convert_pnl(close_pnl, Asset.BTC)
                                if close_pnl_btc != NAN:
                                    cod_pnl_btc = pnl_total_btc - close_pnl_btc

                    value = {
                        'strategy': strategy,
                        'date': date_.strftime('%Y-%m-%d'),
                        'asset': ticker,
                        'position': position_container.size,
                        'avg_price': position_container.entry_price,
                        'live_price': live_price,
                        'exposure_usd': exposure_usd,
                        'pnl_rlzd_usd': realized_pnl_usd,
                        'pnl_rlzd_btc': realized_pnl_btc,
                        'pnl_unrlzd_usd': unrealized_pnl_usd,
                        'pnl_unrlzd_btc': unrealized_pnl_btc,
                        'pnl_cod_usd': cod_pnl_usd,
                        'pnl_cod_btc': cod_pnl_btc,
                        'pnl_total_usd': pnl_total_usd,
                        'pnl_total_btc': pnl_total_btc,
                        'sigma_pnl_usd': sigma_pnl_usd,
                    }

                    image[key] = value
        logging.info('Pushing risksheet update')
        return image

    def _get_historical_closes(self):
        for date_, date_container in self._positions_by_date.iteritems():

            if date_ not in self._historical_closes:
                self._historical_closes[date_] = {}

            for strategy, strategy_container in date_container.iteritems():
                for ticker, position_container in strategy_container.iteritems():

                    if ticker not in self._historical_closes[date_]:
                        close_time = TZ.localize(datetime.combine(date_, CLOSING_TIME))
                        close_price = get_cassandra_trades_ticks(self._cassandra_conn, ticker, close_time)
                        if close_price is not None:
                            self._historical_closes[date_][ticker] = close_price


    def run_bg(self):

        if self._thread_manager:
            thread = thread_manager.create_daemon_thread(target=self.run)
        else:
            thread = Thread(target=self.run)
            thread.daemon = True

        thread.start()

    def run(self):
        self._publisher.start(self._thread_manager)
        self._subscriber.start(self._thread_manager)

        # load snapshots
        for day_diff in range(7):
            date_ = self._current_date - timedelta(days=day_diff + 1)

            snapshot = load_snapshot(
                self._crypto_trading_conn, self._trading_user_id, date_)
            self._positions_by_date[date_] = snapshot

        self._get_historical_closes()

        # initialize publisher
        image = self._create_positions_image()
        self._publisher.add_table(self._risksheet_table_name, image)

        # attach listener
        self._subscriber.add_listener(self)

        # subscribe to the oms
        self._subscriber.subscribe([self._orders_table_name], publisher='crypto_oms')
        self._subscriber.subscribe([self._inputs_table_name], publisher='crypto_server')

        self.ready = True
        self._event.wait()

    def stop(self):
        self._event.set()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - ''%(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Risk sheet publisher')
    parser.add_argument('--trading_user_id', required=True, help='Trading user ID')

    args = parser.parse_args()

    transport = RabbitMQTransport()
    transport.subscription_listener()

    subscriber = TableStreamSubscriber(transport)
    publisher = TableStreamPublisher(transport)

    now_ = datetime.now(tz=TZ)
    cutoff = TZ.localize(datetime.combine(now_.date(), CLOSING_TIME))
    if now_ > cutoff:
        current_date = datetime.today().date() + timedelta(days=1)
    else:
        current_date = datetime.today().date()

    crypto_trading_conn = MongoPersister.init_from_config('crypto_trading', auto_connect=True).db
    cassandra = CassandraConnector.init_from_config(connection_class=LibevConnection, protocol_version=3)
    cassandra.connect(keyspace='contrib')
    cassandra_conn = cassandra.session

    condition = Condition()
    thread_manager = DaemonThreadExceptionManager(condition)
    exceptions_queue = thread_manager.get_queue()

    trading_user_id = ObjectId(args.trading_user_id)
    risk_publisher = RiskSheetPublisher(
        trading_user_id, subscriber, publisher, crypto_trading_conn, cassandra_conn,current_date, thread_manager=thread_manager)

    risk_publisher.run_bg()

    # Block to see if it has any exceptions
    exc_info = exceptions_queue.get(block=True)
    traceback.print_tb(exc_info[2])
    raise exc_info[1]


