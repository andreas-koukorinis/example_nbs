import logging
import datetime
import pandas as pd
from sgmtradingcore.analytics.features.feature import TimeSeriesFeature
from sgmtradingcore.analytics.features.storage import InMemoryFeatureStorage
from sgmtradingcore.backtesting.persistence import MongoStrategyHelper


class BacktestOrders(TimeSeriesFeature):

    REQUIRED_PARAMETERS = {'strategy_name', 'strategy_desc', 'strategy_code', 'trading_user_id',
                  'backtest_kwargs', 'sync_field', 'backtest_kwargs'}

    def __init__(self, runner, params, **kwargs):
        """
        Feature that retrieves basic details from orders placed in a backtest, returns a timeseries
        :param runner:
        :param params: needs strategy_name, strategy_desc, strategy_code, trading_user_id
        backtest_extra_args should be a dict that gets passed to the helper.get_backtest_result_multiple_days
        sync_field: the timestamp in the order that is user to create the timeseries (placed_time / settled_time)
        :param kwargs:
        """
        storage = InMemoryFeatureStorage(self, parent_cache=None)
        super(BacktestOrders, self).__init__(runner, params, storage)

        self.strategy_name = params['strategy_name']
        self.strategy_desc = params['strategy_desc']
        self.strategy_code = params['strategy_code']
        self.trading_user_id = params['trading_user_id']
        self.backtest_kwargs = params['backtest_kwargs']

        # Field of the order to use for alignment (i.e. placed_time or settled_time)
        shared_objects = self._runner.shared_objects()
        self.sync_field = params['sync_field']
        if 'helper' in kwargs:
            self._helper = kwargs['helper']
        else:
            self._helper = MongoStrategyHelper(fixture_cache=shared_objects['fixture_cache'])
        self._fixture_cache = shared_objects['fixture_cache']

    @classmethod
    def _default_parameters(cls, params, runner):
        return {'backtest_kwargs': {}}

    def _get_date_for_fixture(self, fixture_id):
        return self._fixture_cache.get_kickoff(fixture_id)

    def _compute_by_event_id(self, fixture_id, repopulate=False):
        """
        :param repopulate: if True you might need to repopulate the features you depend from
        """
        date = self._get_date_for_fixture(fixture_id)

        logging.info("Fetching backtest orders for {}".format(fixture_id))

        _, orders = self._helper.get_backtest_result_multiple_days(
            self.strategy_name,
            self.strategy_desc,
            self.trading_user_id,
            self.strategy_code,
            str(date.date()),
            str(date.date()),
            **self.backtest_kwargs)

        index = []
        data = []
        for o in orders:
            index.append(o[self.sync_field])
            data.append({'size': o['size'],
                         'size_matched': o['size_matched'],
                         'limit_price': o['price'],
                         'matched_price': o['average_price_matched'],
                         'is_back': o.get('is_back', True),
                         'sticker': o['sticker'],
                         'capital': o['details']['capital']})

        return pd.DataFrame(index=index, data=data)


class LiveTradingOrders(TimeSeriesFeature):

    REQUIRED_PARAMETERS = {'strategy_name', 'strategy_desc', 'trading_user_id', 'sync_field'}

    def __init__(self, runner, params, **kwargs):
        """
        Feature returns basic info from orders placed during live trading
        :param runner:
        :param params: needs strategy_name, strategy_desc and trading_user_id. Also needs sync_field which chooses which
        timestamp in the order should be used to create the timeseries i.e. placed_time or settled_time
        :param kwargs:
        """
        storage = InMemoryFeatureStorage(self, parent_cache=None)
        super(LiveTradingOrders, self).__init__(runner, params, storage)

        self.strategy_name = params['strategy_name']
        self.strategy_desc = params['strategy_desc']
        self.trading_user_id = params['trading_user_id']

        # Field of the order to use for alignment (i.e. placed_time or settled_time)
        shared_objects = self._runner.shared_objects()
        self.sync_field = params['sync_field']
        self._helper = MongoStrategyHelper(fixture_cache=shared_objects['fixture_cache'])
        self._fixture_cache = shared_objects['fixture_cache']

    def _get_date_for_fixture(self, fixture_id):
        return self._fixture_cache.get_kickoff(fixture_id)

    def _compute_by_event_id(self, fixture_id, repopulate=False):
        """
        :param repopulate: if True you might need to repopulate the features you depend from
        """
        date = self._get_date_for_fixture(fixture_id)

        logging.info("Fetching backtest orders for {}".format(fixture_id))

        orders = self._helper.get_prod_orders_between_datetimes(date, date + datetime.timedelta(days=1),
                                                                strategy_name=self.strategy_name,
                                                                strategy_desc=self.strategy_desc,
                                                                trading_user_id=self.trading_user_id)
        index = []
        data = []
        for o in orders:
            index.append(o[self.sync_field])
            data.append({'size': o['size'],
                         'size_matched': o['size_matched'],
                         'limit_price': o['price'],
                         'matched_price': o['average_odds_matched'],
                         'is_back': o['is_back']})

        return pd.DataFrame(index=index, data=data)


class HistoricalOrders(TimeSeriesFeature):
    """
    Uses orders from actual trading if they exist, else get it from a backtest
    """

    BACKTEST_CLASS = BacktestOrders
    LIVE_CLASS = LiveTradingOrders

    REQUIRED_PARAMETERS = {'backtest_kwargs', 'strategy_name', 'strategy_desc', 'strategy_code', 'trading_user_id',
                  'sync_field'}

    def __init__(self, runner, params, **kwargs):
        """
        This class returns orders placed during live trading for an event, or if that hasn't happened it gets orders
        from a backtest.
        :param runner:
        :param params: strategy_name, strategy_desc, strategy_code, trading_user_id
        sync_field: field of the order used to create the timeseries (placed_time / settled_time)
        :param kwargs:
        """
        storage = InMemoryFeatureStorage(self, parent_cache=None)

        super(HistoricalOrders, self).__init__(runner, params, storage)

        self.backtest_kwargs = params['backtest_kwargs']
        self.strategy_name = params['strategy_name']
        self.strategy_desc = params['strategy_desc']
        self.strategy_code = params['strategy_code']
        self.trading_user_id = params['trading_user_id']
        self.sync_field = params['sync_field']

    def _live_params(self):
        return {'strategy_name': self.strategy_name,
                'strategy_desc': self.strategy_desc,
                'trading_user_id': self.trading_user_id,
                'sync_field': self.sync_field}

    def _backtest_params(self):
        live_params = self._live_params()
        live_params.update({'strategy_code': self.strategy_code,
                            'backtest_kwargs': self.backtest_kwargs})
        return live_params

    def _compute_by_event_id(self, event_id, repopulate=False):
        """
        :param repopulate: if True you might need to repopulate the features you depend from
        """
        logging.info("Fetching orders for {}, {}".format(self.feature_id, event_id))

        from sgmtradingcore.analytics.features.request import FeatureRequest
        live_orders_request = FeatureRequest(self.LIVE_CLASS.__name__, self._live_params(), {})
        live_orders = self._runner.get_dataframes_by_event_ids([live_orders_request], [event_id])[event_id]
        if not live_orders.empty:
            return live_orders

        backtest_orders_request = FeatureRequest(self.BACKTEST_CLASS.__name__, self._backtest_params(), {})
        backtest_orders = self._runner.get_dataframes_by_event_ids([backtest_orders_request], [event_id])[event_id]
        return backtest_orders


if __name__ == '__main__':
    from sgmtradingcore.analytics.features.runner import FeatureRunner
    from sgmtradingcore.analytics.features.request import FeatureRequest
    from sgmtradingcore.strategies.realtime import get_environment
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - ''%(levelname)s - %(message)s')

    event_ids = ['ENP2614510', 'ENP2614512', 'ENP2614513', 'ENP2614514', 'ENP2614511', 'ENP2614515']

    requests = [
        FeatureRequest('HistoricalOrders', {'strategy_name': 'bball_pbp',
                                            'strategy_desc': 'nba_deadball_lambda_1',
                                            'strategy_code': 'nba_quarters_db',
                                            'trading_user_id': get_environment('PROD')['trading_user_id'],
                                            'backtest_kwargs': {'mnemonic': 'nba_lambda_prod_quarters_3'},
                                            'sync_field': 'placed_time'}, {}),
        FeatureRequest('BacktestOrders', {'strategy_name': 'bball_pbp',
                                          'strategy_desc': 'nba_deadball_lambda_1',
                                          'strategy_code': 'nba_quarters_db',
                                          'trading_user_id': get_environment('PROD')['trading_user_id'],
                                          'backtest_kwargs': {'mnemonic': 'nba_lambda_prod_quarters_3'},
                                          'sync_field': 'placed_time'}, {}),
        FeatureRequest('LiveTradingOrders', {'strategy_name': 'bball_pbp',
                                             'strategy_desc': 'nba_deadball_lambda_1',
                                             'strategy_code': 'nba_quarters_db',
                                             'trading_user_id': get_environment('PROD')['trading_user_id'],
                                             'sync_field': 'placed_time'}, {})
    ]

    runner = FeatureRunner()
    df = runner.get_dataframes_by_event_ids(requests, event_ids)

    print df
