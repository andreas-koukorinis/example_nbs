import pytz
from datetime import date, datetime
from sgmtradingcore.backtesting.persistence import MongoStrategyHelper
from stratagemdataprocessing.parsing.common.stickers import BOOKMAKER_ABBR
from stratagemdataprocessing.parsing.common.stickers import parse_sticker
from stratagemdataprocessing.events.fixture_cache import FixtureCache
from sgmtradingcore.util.misc import daterange
from sgmtradingcore.core.trading_types import OrderStatus


class RunStats(object):

    def __init__(self, **kwargs):

        # info about the run
        self.mnemonic = None

        # {event_id: float}
        self.per_event_pnl = kwargs.get('per_event_pnl', {})

        # {event_id: float}
        self.per_event_vol = {}

        # {date: pnl}
        self.pnl_by_date = {}

        # [pnl]
        self.daily_cumulative_pnl = []

        # {bm_abbr: float}
        self.pnl_by_bookmaker = {}

        # {bm_abbr: float} matched vol by bm
        self.vol_by_bookmaker = {}

        self.pnl_by_side = {
            'back': 0.,
            'lay': 0.
        }


class StrategyRunStatsHelper(object):
    """
    To load stats about one strategy run:
    1) load_orders_and_instructions()
    2) get_stats()
    3) remove instructions and orders from memory
    4) use sgmtradingcore/analytics/comparison/postprocess_stats.py to analise the stats
    5) use sgmtradingcore/analytics/comparison/presentation.py to create charts

    """
    def __init__(self, helper=None, fixture_cache=None):
        self._fixture_cache = fixture_cache or FixtureCache()
        self._helper = helper or MongoStrategyHelper(fixture_cache=self._fixture_cache)

    def load_orders_and_instructions(self, strategy_name, strategy_desc, strategy_code, mnemonic, trading_user_id,
                                     start_date, end_date, is_prod=False):
        """
        :param trading_user_id: like 'trading_user_id'
        :param start_date: datetime.date
        :return:
        """

        start_date_str = start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d')
        end_date_str = end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d')
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        if is_prod:
            instructions, orders = self._helper.get_prod_instructions_and_orders_between_fixture_datetimes(
                strategy_name=strategy_name,
                strategy_desc=strategy_desc,
                strategy_code=strategy_code,
                trading_user_id=trading_user_id,
                start_dt=datetime.combine(start_date, datetime.min.time()).replace(tzinfo=pytz.utc),
                end_dt=datetime.combine(end_date, datetime.max.time()).replace(tzinfo=pytz.utc)
            )
        else:
            instructions, orders = self._helper.get_backtest_result_multiple_days(strategy_name=strategy_name,
                                                                                  strategy_desc=strategy_desc,
                                                                                  strategy_code=strategy_code,
                                                                                  mnemonic=mnemonic,
                                                                                  trading_user_id=trading_user_id,
                                                                                  start_date=start_date_str,
                                                                                  end_date=end_date_str)

        return instructions, orders

    def get_stats(self, strategy_name, strategy_desc, strategy_code, mnemonic, trading_user_id, start_date, end_date,
                  instructions, orders):
        """
        :param trading_user_id: like 'trading_user_id'
        :param start_date: datetime.date
        :return:
        """

        stats = RunStats()
        stats.mnemonic = mnemonic

        start_date_str = start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d')
        end_date_str = end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d')
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        per_event_pnl = self.get_per_event_pnl(stats, instructions, orders)
        stats.per_event_pnl = per_event_pnl

        per_event_vol = self.get_per_event_vol(stats, instructions, orders)
        stats.per_event_vol = per_event_vol

        pnl_by_date = self.get_pnl_by_date(stats, instructions, orders, start_date, end_date)
        stats.pnl_by_date = pnl_by_date

        daily_cumulative_pnl = self.get_daily_cumulative_pnl(stats)
        stats.daily_cumulative_pnl = daily_cumulative_pnl

        pnl_by_bookmaker = self.get_pnl_by_bookmaker(stats, instructions, orders, start_date, end_date)
        stats.pnl_by_bookmaker = pnl_by_bookmaker

        vol_by_bookmaker = self.get_vol_by_bookmaker(stats, instructions, orders, start_date, end_date)
        stats.vol_by_bookmaker = vol_by_bookmaker

        pnl_by_side = self.get_pnl_by_side(stats, instructions, orders, start_date, end_date)
        stats.pnl_by_side = pnl_by_side

        return stats

    def get_pnl_by_side(self, stats, instructions, orders, start_date, end_date):
        pnl_by_side = {
            'back': 0.,
            'lay': 0.
        }

        for order in orders:
            pnl = self.get_order_pnl(order)
            is_back = self.get_order_side(order)
            if is_back:
                pnl_by_side['back'] += pnl
            else:
                pnl_by_side['lay'] += pnl

        for key in pnl_by_side.keys():
            pnl_by_side[key] = round(pnl_by_side[key], 2)
        return pnl_by_side

    def get_pnl_by_bookmaker(self, stats, instructions, orders, start_date, end_date):
        pnl_by_bookmaker = {}

        for order in orders:
            pnl = self.get_order_pnl(order)
            bm = self.get_order_bm(order)

            if bm not in pnl_by_bookmaker:
                pnl_by_bookmaker[bm] = 0.
            pnl_by_bookmaker[bm] += pnl

        for key in pnl_by_bookmaker.keys():
            pnl_by_bookmaker[key] = round(pnl_by_bookmaker[key], 2)
        return pnl_by_bookmaker

    def get_vol_by_bookmaker(self, stats, instructions, orders, start_date, end_date):
        vol_by_bookmaker = {}

        for order in orders:
            bm = self.get_order_bm(order)
            if bm not in vol_by_bookmaker:
                vol_by_bookmaker[bm] = 0.
            vol = self.get_order_matched_vol(order)
            vol_by_bookmaker[bm] += vol

        for key in vol_by_bookmaker.keys():
            vol_by_bookmaker[key] = round(vol_by_bookmaker[key], 2)
        return vol_by_bookmaker

    def get_pnl_by_date(self, stats, instructions, orders, start_date, end_date):

        pnl_by_date = {}
        for day in daterange(start_date, end_date):
            pnl_by_date[day] = 0.
        for event_id, pnl in stats.per_event_pnl.iteritems():
            kickoff_date = self._fixture_cache.get_kickoff(event_id).date()
            pnl_by_date[kickoff_date] += pnl

        for key in pnl_by_date.keys():
            pnl_by_date[key] = round(pnl_by_date[key], 2)
        return pnl_by_date

    def get_per_event_pnl(self, stats, instructions, orders):
        per_event_pnl = {}

        for order in orders:
            pnl = StrategyRunStatsHelper.get_order_pnl(order)
            if pnl == 0.:
                continue
            sport, (scope, event_id), mkt, params, _ = parse_sticker(order['sticker'])
            if event_id not in per_event_pnl:
                per_event_pnl[event_id] = 0.
            per_event_pnl[event_id] += pnl

        for key in per_event_pnl.keys():
            per_event_pnl[key] = round(per_event_pnl[key], 2)
        return per_event_pnl

    def get_per_event_vol(self, stats, instructions, orders):
        per_event_vol = {}

        for order in orders:
            pnl = StrategyRunStatsHelper.get_order_matched_vol(order)
            if pnl == 0.:
                continue
            sport, (scope, event_id), mkt, params, _ = parse_sticker(order['sticker'])
            if event_id not in per_event_vol:
                per_event_vol[event_id] = 0.
            per_event_vol[event_id] += pnl

        for key in per_event_vol.keys():
            per_event_vol[key] = round(per_event_vol[key], 2)
        return per_event_vol

    def get_daily_cumulative_pnl(self, stats):
        cumulative_pnl = 0.
        daily_cumulative_pnl = []
        for day in sorted(stats.pnl_by_date.keys()):
            cumulative_pnl += stats.pnl_by_date[day]
            daily_cumulative_pnl.append(cumulative_pnl)

        return daily_cumulative_pnl

    @staticmethod
    def get_order_bm(order):
        sport, (scope, event_id), mkt, params, bm = parse_sticker(order['sticker'])
        if bm is not None:
            return BOOKMAKER_ABBR[bm]
        return order['execution_details']['bookmaker']

    @staticmethod
    def get_order_pnl(order):
        if order['status'] != OrderStatus.SETTLED:
            return 0.
        rate = 1.0
        if 'exchange_rate' in order and 'currency' in order and order['exchange_rate'] != 0:
            rate = order['exchange_rate']

        if 'pnl' in order:
            return order['pnl'] / rate
        else:
            if 'outcome' in order.keys():
                return order['outcome']['gross'] / rate
            else:
                return 0.

    @staticmethod
    def get_order_side(order):
        return order['bet_side'] == 'back'

    @staticmethod
    def get_order_matched_vol(order):
        if order['status'] != OrderStatus.SETTLED:
            return 0.
        return order['size_matched']
