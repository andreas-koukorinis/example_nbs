import numbers
from sgmtradingcore.providers.crypto.market.lob_provider import HistoricalLobProvider
from sgmtradingcore.analytics.performance import get_trading_days
from sgmtradingcore.core.definitions import tsrun
from sgmtradingcore.core.engines import SimulationEngine
from sgmtradingcore.core.miscellaneous import (
    DelayedTickPush, EmptyProvider, DelayedTickPusher, MultiScalarMultiInputCombiner, empty_provider)
from sgmtradingcore.exchange.exchange_sim import ExchangeSimulator, MatchingMode
from sgmtradingcore.exchange.trades_matcher.trades_matcher import WiredTradesMatcher, TradesMatcher
from sgmtradingcore.exchange.trades_matcher.trades_filler import TradesFiller
from sgmtradingcore.exchange.trades_matcher.crypto_trades_filler import CryptoTradesFiller
from sgmtradingcore.execution.algo_manager import AlgoManager
from sgmtradingcore.execution.controls import ControlsApplicator
from sgmtradingcore.providers.action.football.cleanfeed_providers import HistoricalCleanfeedFootballActionDataProvider
from sgmtradingcore.providers.action.tennis.tennis_action_providers import HistoricalTennisActionDataProvider, HistoricalLsportsTennisActionDataProvider
from sgmtradingcore.providers.action.basketball.basketball_action_provider import HistoricalBasketballActionDataProvider
from sgmtradingcore.providers.model.sporterpilot.football_model_backtest import (
    SporterPilotFootballBacktestModelProvider)
from sgmtradingcore.providers.odds_providers import (
    HistoricalFileOddsProvider, DemuxApplicator, DefaultDemux)
from sgmtradingcore.providers.refdata_providers import (
    TennisReferenceDataProvider,
    FootballReferenceDataProvider,
    BasketballReferenceDataProvider,
    CryptoReferenceDataProvider)
from sgmtradingcore.providers.risk.risk_pnl import HistoricalRiskProvider
from sgmtradingcore.strategies.realtime import FrameworkRealtimeProviders
from sgmtradingcore.providers.crypto.trades.trades_provider import HistoricalCryptoTradesProvider
from sgmtradingcore.providers.crypto.risk.risk_provider import HistoricalCryptoRiskProvider
from stratagemdataprocessing.dbutils.mongo import MongoPersister
from stratagemdataprocessing.enums.odds import Sports, Bookmakers, Providers
from sgmtradingcore.util.misc import daterange
from sgmtradingcore.crypto.asset_inventory import AssetsInventoryManager


class FrameworkHistoricalProviders(object):

    @staticmethod
    def action_provider(sport, delay_ms=0, provider=None):
        if sport == Sports.TENNIS:
            if provider is None or provider == Providers.ENET:
                action_prov = HistoricalTennisActionDataProvider(delay_ms=delay_ms, strict=False)
            elif provider == Providers.LSPORTS:
                action_prov = HistoricalLsportsTennisActionDataProvider(delay_ms=delay_ms, strict=False)
            else:
                raise ValueError("Bad provider {}".format(provider))
        elif sport == Sports.FOOTBALL:
            mongo_odds = MongoPersister.init_from_config('odds', auto_connect=True).db
            # mongo_football = MongoPersister.init_from_config('football', auto_connect=True).db
            action_prov = HistoricalCleanfeedFootballActionDataProvider(mongo_odds, delay_ms=delay_ms)
            # action_prov = HistoricalFootballCrowdScoresActionDataProvider(mongo_odds, mongo_football)
        elif sport == Sports.BASKETBALL:
            if delay_ms > 0:
                raise NotImplementedError("delay_ms not implemented for BASKETBALL")
            return HistoricalBasketballActionDataProvider()
        elif sport == Sports.CRYPTO:
            if delay_ms > 0:
                raise NotImplementedError("delay_ms not implemented for CRYPTO")
            return EmptyProvider()
        else:
            raise ValueError('Cannot create action provider for sport %s' % sport)

        return action_prov

    @staticmethod
    def refdata_provider(sport):
        if sport == Sports.TENNIS:
            refdata_prov = TennisReferenceDataProvider()
        elif sport == Sports.FOOTBALL:
            refdata_prov = FootballReferenceDataProvider()
        elif sport == Sports.BASKETBALL:
            refdata_prov = BasketballReferenceDataProvider()
        elif sport == Sports.CRYPTO:
            refdata_prov = CryptoReferenceDataProvider()
        else:
            raise ValueError('Cannot create RefData provider for sport %s' % sport)

        return refdata_prov

    @staticmethod
    def market_provider(sport, odds_cache=None, expiration_ms=0, trades_cache=None):
        if sport in (Sports.TENNIS, Sports.FOOTBALL, Sports.BASKETBALL):
            refdata_prov = HistoricalFileOddsProvider(odds_cache=odds_cache, odds_expiration_ms=expiration_ms)
        elif sport == Sports.CRYPTO:
            refdata_prov = HistoricalLobProvider(trades_cache=trades_cache, expiration_ms=expiration_ms)
        else:
            raise ValueError('Cannot create market provider for sport %s' % sport)

        return refdata_prov

    @staticmethod
    def trades_provider(sport):
        if sport in (Sports.TENNIS, Sports.FOOTBALL, Sports.BASKETBALL):
            refdata_prov = EmptyProvider()
        elif sport == Sports.CRYPTO:
            refdata_prov = HistoricalCryptoTradesProvider()
        else:
            raise ValueError('Cannot create trades provider for sport %s' % sport)

        return refdata_prov

    @staticmethod
    def model_provider(sport, model_name="no_model", model_opt_name=None, model_config=None):
        """
        A model is identified by a model_name.
        A model take a configuration in input e.g. model_config, however we can instead specify a model_opt_name which
        identify a specific configuration for the model.
        :param sport:
        :param model_name: str, identify the model
        :param model_opt_name: str, identify the model input parameters.
        :param model_config: input of the model, alternative to model_opt_name
        :return:
        """
        if model_opt_name is not None and model_config is not None:
            raise ValueError("Specify only one of two model_opt_name, model_config")
        if sport == Sports.TENNIS:
            if model_name == "tennis_emd":
                from sgmtradingcore.providers.model.emd_inplay_model import TennisEMDInPlayModelProvider

                model_prov = TennisEMDInPlayModelProvider(model_opt_name=model_opt_name, options=model_config)

            else:
                model_prov = FrameworkRealtimeProviders.model_provider(Sports.TENNIS, model_name,
                                                                       model_opt_name=model_opt_name, model_config=model_config)

        elif sport == Sports.FOOTBALL:
            if model_name == 'sporterpilot':
                model_prov = SporterPilotFootballBacktestModelProvider()
            else:

                from sgmfootball.trading.providers.model.inplay_model import (
                    HistoricalFootballInPlayModelProvider,
                    DEFAULT_CACHE_DIR)
                if model_name == "factor_model":
                    raise NotImplementedError("Need to provide it from sgmfootball repo")
                else:
                    if model_name == "no_model":
                        model_name = "football_in_play"
                        model_opt_name = "default"
                    model_prov = HistoricalFootballInPlayModelProvider(model_name, model_opt_name, cache_dir=DEFAULT_CACHE_DIR)
        elif sport == Sports.BASKETBALL:
            model_prov = EmptyProvider()
        elif sport == Sports.CRYPTO:
            model_prov = EmptyProvider()
        else:
            raise ValueError('Cannot create model provider for sport %s' % sport)

        return model_prov

    @staticmethod
    def risk_provider(strategy_descrs, sport, sport_filter, start_dt,
                      end_dt, initial_capital, inventory_manager=None):
        """

        :param strategy_descrs: list of all the possible strategy decriptions that this strategy can use
        :param sport:
        :param sport_filter:
        :param start_dt:
        :param end_dt:
        :param initial_capital:
        :param inventory_manager: AssetsInventoryManager
        :return:
        """

        if sport == Sports.CRYPTO:
            risk_prov = HistoricalCryptoRiskProvider(strategy_descrs, inventory_manager)
        else:
            if isinstance(initial_capital, numbers.Number):
                initial_capital = {descr: {start_dt.date(): initial_capital} for descr in strategy_descrs}

            risk_prov = HistoricalRiskProvider(strategy_descrs, initial_capital)

        return risk_prov

    @staticmethod
    def algo_manager():
        algo_manager = AlgoManager(send_notifications=False)
        return algo_manager


def run_backtest(strategy, start_time, end_time=None,
                 refdata_provider=None, odds_provider=None, trades_provider=None, action_provider=None,
                 cmd_provider=None, model_provider=None, risk_provider=None, algo_manager=None,
                 matching_mode=MatchingMode.PRICE_MATCH_ENTIRE_SIZE,
                 initial_capital=10000, trading_days=None, total=None, initial_orders=None,
                 bookmakers=None, exchange_sim_kwargs=None, delay_bets_ms=0, action_delay_ms=0,
                 framework_providers=FrameworkHistoricalProviders,
                 record_trading_signals=True, record_trade_intentions=False,
                 use_trades_matcher=False, trade_filler_kwargs=None, inventory_manager=None):
    """
    Run locally a backtest simulation
    :return: a dict of values. Specifically 'bet_states' type is [{datatime.datatime: Order}] and contains
        update time (not placed_time) and the orders.
    :param use_trades_matcher: use TradesMatched instead of ExchangeSimulator
    :param inventory_manager: AssetsInventoryManager
    :param trade_filler_kwargs: kwargs for TradesFiller
    """

    def circuit():
        """
        Wires the various providers, a strategy and exchange simulator to run a backtest.

        """
        refdata_prov = refdata_provider or framework_providers.refdata_provider(strategy.get_sport())
        strategy.set_refdata_provider(refdata_prov)

        market_provider = odds_provider or framework_providers.market_provider(strategy.get_sport())
        strategy.set_market_provider(market_provider)

        trades_prov = trades_provider or framework_providers.trades_provider(strategy.get_sport())
        strategy.set_trades_provider(trades_prov)

        action_prov = action_provider or framework_providers.action_provider(strategy.get_sport(),
                                                                             delay_ms=action_delay_ms)
        strategy.set_action_provider(action_prov)

        model_prov = model_provider or framework_providers.model_provider(strategy.get_sport())
        strategy.set_model_provider(model_prov)

        if cmd_provider is not None:
            commands = cmd_provider()['commands']
        else:
            commands = None

        risk_prov = risk_provider or framework_providers.risk_provider(
            strategy.strategy_run_ids,
            strategy.get_sport(),
            strategy.get_sport_filter(),
            start_time, end_time,
            initial_capital,
            inventory_manager=inventory_manager)

        market_provider_output = market_provider()
        trades_provider_output = trades_prov()
        refdata_provider_output = refdata_prov()
        action_provider_output = action_prov()
        model_provider_output = model_prov()

        order_push_port = DelayedTickPush()  # to work around the loop in the graph
        delayed_order_states = order_push_port()
        order_pusher = DelayedTickPusher(order_push_port)

        instruction_push_port = DelayedTickPush()  # to work around the loop in the graph
        delayed_instruction_states = instruction_push_port()
        instruction_pusher = DelayedTickPusher(instruction_push_port)

        risk_provider_output = risk_prov(delayed_instruction_states, delayed_order_states)

        if strategy.use_algo_manager():
            algo_state_push_port = DelayedTickPush()  # to work around the loop in the graph
            delayed_algo_states = algo_state_push_port()

            strategy_top_level_wiring = strategy.top_level_wiring()
            strategy_outputs = strategy_top_level_wiring(
                delayed_instruction_states, delayed_order_states, risk_provider_output, commands,
                algo_states=delayed_algo_states)

            algo_man = algo_manager or framework_providers.algo_manager()
            algo_man.set_refdata_provider(refdata_prov)
            algo_man.set_market_provider(market_provider)
            algo_man.set_action_provider(action_prov)
            algo_man.set_model_provider(model_prov)

            algo_manager_wiring = algo_man.top_level_wiring()
            algo_manager_outputs = algo_manager_wiring(
                delayed_instruction_states, delayed_order_states, risk_provider_output, strategy_outputs['bet_transactions'])

            algo_state_pusher = DelayedTickPusher(algo_state_push_port)
            algo_state_pusher_output = algo_state_pusher(algo_manager_outputs['algo_states'])

            analysis_combiner = MultiScalarMultiInputCombiner()
            analysis_output = analysis_combiner(strategy_outputs['analysis'], algo_manager_outputs['analysis'])

            transactions_out = algo_manager_outputs['bet_transactions']

        else:
            strategy_top_level_wiring = strategy.top_level_wiring()
            strategy_outputs = strategy_top_level_wiring(
                delayed_instruction_states, delayed_order_states, risk_provider_output, commands)

            delayed_algo_states = empty_provider()
            algo_state_pusher_output = empty_provider()

            analysis_output = strategy_outputs['analysis']

            transactions_out = strategy_outputs['bet_transactions']

        strategy_controls = strategy.get_downstream_controls()
        controller = ControlsApplicator([strategy_controls], None)
        controlled_transactions = controller(transactions_out, risk_provider_output)

        bookms = bookmakers or [Bookmakers.BETFAIR]

        if not use_trades_matcher:
            exchange_sim_kw = exchange_sim_kwargs or {}

            exchange = ExchangeSimulator(
                market_provider, action_prov, refdata_prov,
                matching_mode=matching_mode, bookmakers=bookms, delay_bets_ms=delay_bets_ms, **exchange_sim_kw
            )

            exchange_refdata = exchange.get_refdata_push_port()()
            exchange_market_data = DemuxApplicator(DefaultDemux())(exchange.get_market_push_port()())
            exchange_action_data = exchange.get_action_push_port()()
            exchange_outputs = exchange(
                controlled_transactions['allowed'], exchange_market_data, exchange_refdata, exchange_action_data)
        else:

            trade_filler_kwargs_ = trade_filler_kwargs or {}
            if strategy.get_sport() == Sports.CRYPTO:
                trades_filler = CryptoTradesFiller(market_provider.get_trades_cache(), inventory_manager,
                                                   **trade_filler_kwargs_)
            else:
                trades_filler = TradesFiller(market_provider.get_odds_cache(), **trade_filler_kwargs_)
            matcher = TradesMatcher(trades_filler=trades_filler)
            exchange = WiredTradesMatcher(refdata_prov, matcher=matcher, delay_bets_ms=delay_bets_ms)

            exchange_refdata = exchange.get_refdata_push_port()()

            exchange_outputs = exchange(
                controlled_transactions['allowed'], exchange_refdata)

        order_pusher_output = order_pusher(exchange_outputs['orders'])
        instruction_pusher_output = instruction_pusher(exchange_outputs['instructions'])

        circuit_result = {
            'analysis': analysis_output,
            'orders': delayed_order_states,  # note we output the delayed by 1ms version, ie what the strategy sees
            'order_pusher_output': order_pusher_output,  # to avoid GC
            'instructions': delayed_instruction_states,  # note we output the delayed by 1ms version, ie what the strategy sees
            'instruction_pusher_output': instruction_pusher_output,  # to avoid GC
            'algo_state_pusher_output': algo_state_pusher_output, # to avoid GC
            'odds_provider': market_provider_output,
            'trades_provider': trades_provider_output,
            'action_provider': action_provider_output,
            'refdata_provider': refdata_provider_output,
            'model_provider': model_provider_output,
            'risk_provider': risk_provider_output,
            'rejected_transactions': controlled_transactions['rejected'],
            'algo_states': delayed_algo_states,
        }

        if record_trading_signals and 'trading_signals' in strategy_outputs:
            signals_output = strategy_outputs['trading_signals']
            circuit_result.update({'trading_signals': signals_output})
        if record_trade_intentions and 'trade_intentions' in strategy_outputs:
            trade_intentions_output = strategy_outputs['trade_intentions']
            circuit_result.update({'trade_intentions': trade_intentions_output})

        return circuit_result

    recorders_options = {
        'risk_provider': (True, True),
    }

    engine = SimulationEngine()
    result = tsrun(circuit, engine, start_time, end_time, record_outputs=True, recorders_options=recorders_options)
    return result
