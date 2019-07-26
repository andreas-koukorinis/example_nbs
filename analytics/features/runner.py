import time
import logging
import traceback
from Queue import Queue
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from multiprocessing.util import Finalize
from threading import Thread
import os

import psutil
from pymongo.errors import DuplicateKeyError

from sgmtradingcore.analytics.features.bball_feature import BBallABCBestOdds, BBallOddsCacheBestOdds, BBallPreLineupOdds
from sgmtradingcore.analytics.features.fball_features.odds import FballABCBestOdds
from sgmtradingcore.analytics.features.feature import EventFeature, TimeSeriesFeature, \
    InfiniteTimeSeriesFeature, merge_timeseries_features
from sgmtradingcore.analytics.features.market_indicators import AvailableStickers, AvailableStickersSingleBookmaker
from sgmtradingcore.analytics.features.market_indicators import BookPressure, MicroPrice
from sgmtradingcore.analytics.features.mean_reversion_indicators import MeanReversionIndicators
from sgmtradingcore.analytics.features.request import FeatureRequest
from sgmtradingcore.analytics.features.trading_history import BacktestOrders, HistoricalOrders, LiveTradingOrders
from sgmtradingcore.analytics.features.vpin import VPINBasicFeature, VPINBasicTSFeature
from sgmtradingcore.analytics.features.volume_buckets_feature import VolumeBucketFeature
from sgmtradingcore.analytics.features.lob_feature import LobOrdersFeature
from sgmtradingcore.util.misc import chunks
from stratagemdataprocessing.bookmakers.common.odds.cache import HistoricalCassandraOddsCache
from stratagemdataprocessing.dbutils.mongo import MongoPersister
from stratagemdataprocessing.dbutils.mysql import MySQLClient
from stratagemdataprocessing.enums.odds import Bookmakers
from stratagemdataprocessing.events.fixture_cache import FixtureCache
from stratagemdataprocessing.parsing.common.cassandra_dump import get_cassandra_connection

FEATURE_CLASSES = [BookPressure,
                   MicroPrice,
                   HistoricalOrders,
                   LiveTradingOrders,
                   BacktestOrders,
                   BBallPreLineupOdds,
                   BBallOddsCacheBestOdds,
                   BBallABCBestOdds,
                   FballABCBestOdds,
                   AvailableStickers,
                   AvailableStickersSingleBookmaker,
                   MeanReversionIndicators,
                   VPINBasicFeature,
                   VPINBasicTSFeature,
                   VolumeBucketFeature,
                   LobOrdersFeature
                   ]

# Multi processing functions
parallel_runner = None


def init_parallel_runner(*args):
    """
    This initializes a new runner in each child process that calls it. We need to make the runner global so we can
    access it from compute_for_sticker_mp / compute_for_event_mp
    Clean up from:
    https://stackoverflow.com/questions/24717468/context-managers-and-multiprocessing-pools/24724452#24724452
    :param args:
    :return:
    """
    global parallel_runner
    parallel_runner = args[0](extra_feature_classes=args[1], **args[2])

    try:
        current_niceness = os.nice(0)
        os.nice(current_niceness - 10)
    except:
        pass

    Finalize(object, clean_up_mp, exitpriority=16)


def clean_up_mp(*args, **kwargs):
    """
    Cleans up connections when processes are torn down
    :param args:
    :param kwargs:
    :return:
    """
    global parallel_runner
    parallel_runner.cleanup()


def compute_for_event_mp(in_args):
    """
    For time series features calculated on an event-by-event basis
    :param in_args:
    :return:
    """
    feature_requests, events = in_args
    global parallel_runner
    parallel_runner.compute_dataframes_by_event_id(feature_requests, events)


def compute_event_features_mp(in_args):
    """
    For event based features (not timeseries)
    :param in_args:
    :return:
    """
    feature_requests, event_ids = in_args
    global parallel_runner
    parallel_runner.compute_event_features(feature_requests, event_ids)


def compute_for_sticker_mp(in_args):
    """
    Only for Timeseries features - used for processing stickers in batches in different processes
    :param in_args:
    :return:
    """
    feature_requests, stickers = in_args
    global parallel_runner
    parallel_runner.compute_dataframes_by_stickers(feature_requests, stickers)


def compute_crypto_tickers_features_mp(in_args):
    """
    Only for InfiniteTimeseries features - used for processing tone ticker for different timeranges
    :param in_args:
    :return:
    """
    try:
        feature_requests, start_dt, end_dt = in_args
        global parallel_runner
        parallel_runner.compute_dataframes(feature_requests, start_dt, end_dt)
    except Exception as e:
        logging.error("Error in worker process: {}".format(e))
        logging.error("{}".format(traceback.format_exc()))
        raise


def compute_for_sticker_q(feature_requests, stickers, feature_map, result_queue):
    ret = dict()
    for sticker in stickers:
        features_df = list()
        for request in feature_requests:
            feature_id = request.feature_id
            # logging.info("Requesting {} {}".format(feature_id, sticker))
            features_df.append(feature_map[feature_id].get_df_from_sticker(sticker))
        ret[sticker] = merge_timeseries_features(features_df)
    result_queue.put(ret)


class BaseFeatureRunner(object):
    """
    Keep all Feature objects, one object per feature_id
    """

    def __init__(self, extra_feature_classes=None, **kwargs):
        """

        :param extra_feature_classes:
        :param kwargs:
            cassandra_connection:
            fixture_cache:
            odds_cache:
            odds_cache_parse_false:
            mysql_client:
        """
        self._FEATURE_CLASSES = FEATURE_CLASSES

        self._extra_feature_classes = extra_feature_classes or []
        self._FEATURE_CLASSES = list(set(self._FEATURE_CLASSES + self._extra_feature_classes))
        self._CLASSES_MAP = {c.__name__: c for c in self._FEATURE_CLASSES}
        self.feature_map = dict()  # {feature_id: Feature}
        self._kwargs = kwargs

        names = {c.__name__ for c in self._FEATURE_CLASSES}
        if len(names) != len(self._FEATURE_CLASSES):
            raise ValueError("Duplicate class names")

        self.SHARED_OBJECTS = dict()
        self._init_shared_objs()

    def classes_map(self):
        return self._CLASSES_MAP

    @classmethod
    def merge_timeseries_features(cls, dfs):
        return merge_timeseries_features(dfs)

    def _init_shared_objs(self):
        raise NotImplementedError()

    def feature_conn(self):
        return self.SHARED_OBJECTS.get('feature_conn')

    def cleanup(self):
        raise NotImplementedError()

    def shared_objects(self):
        return self.SHARED_OBJECTS

    def initialize_mp_pool(self, n_jobs, initializer=None, init_args=None, maxtasksperchild=1):
        if initializer is None:
            initializer = init_parallel_runner

        if init_args is None:
            init_args = (self.__class__, self._extra_feature_classes, self._kwargs)

        return Pool(processes=n_jobs, initializer=initializer, initargs=init_args, maxtasksperchild=maxtasksperchild)

    def _precompute(self, feature_requests, in_args, n_jobs=None, batch_size=50, input_type='stickers'):

        """
        Pre-computes feature with multiprocessing and doesn't return.

        :param feature_requests: Only Timeseries features
        :param stickers: list of stickers to compute for
        :param n_jobs: number of processes to start. If None starts n_cores processes in the pool
        :param batch_size: how many stickers should be processed on each iteration. (May be altered if len(stickers) is
        small
        :return: None
        """

        if input_type == 'stickers':
            target_function = compute_for_sticker_mp
            target_class = TimeSeriesFeature
        elif input_type == 'event_ids':
            target_function = compute_for_event_mp
            target_class = TimeSeriesFeature
        elif input_type == 'event_features':
            target_function = compute_event_features_mp
            target_class = EventFeature
        else:
            raise ValueError('Unknown precompute input_type {}'.format(input_type))

        from tqdm import tqdm

        if n_jobs is None:
            n_jobs = cpu_count()

        if not isinstance(feature_requests, list):
            feature_requests = [feature_requests]

        self._instantiate_features(feature_requests)

        if input_type == 'stickers':
            for request in feature_requests:
                feature_id = request.feature_id
                self.feature_map[feature_id].initialize_stickers(in_args)

        feature_requests = [f for f in feature_requests if
                            isinstance(self.feature_map[f.feature_id], target_class)]

        if len(feature_requests) == 0:
            logging.info('No {} to calculate'.format(target_class.__name__))
            return

        batch_size = int(min(batch_size, ((len(in_args) + n_jobs - 1) / n_jobs)))
        stickers_batches = [c for c in chunks(in_args, batch_size)]

        p = self.initialize_mp_pool(n_jobs)

        input_args = [(feature_requests, s) for s in stickers_batches]

        pbar = tqdm(total=len(input_args) * batch_size)

        def update(*args):
            pbar.update(batch_size)

        for arg in input_args:
            p.apply_async(target_function, args=(arg,), callback=update)

        p.close()
        p.join()

    def _precompute_infinite_timeseries(self, feature_requests, start_dt, end_dt, n_jobs=None, input_type='tickers'):

        """
        Pre-computes feature with multiprocessing and doesn't return.

        :param feature_requests: Only InfiniteTimeseries features
        :param n_jobs: number of processes to start. If None starts n_cores processes in the pool
        :param batch_range: (timedelta) which range should each worker take care of
        :return: None
        """

        if input_type == 'tickers':
            target_function = compute_crypto_tickers_features_mp
            target_class = InfiniteTimeSeriesFeature
        else:
            raise ValueError('Unknown precompute input_type {}'.format(input_type))

        from tqdm import tqdm

        if n_jobs is None:
            n_jobs = cpu_count()

        if not isinstance(feature_requests, list):
            feature_requests = [feature_requests]

        self._instantiate_features(feature_requests)

        if input_type == 'tickers':
            for request in feature_requests:
                feature_id = request.feature_id
                self.feature_map[feature_id].initialize(start_dt, end_dt)

        feature_requests = [f for f in feature_requests if
                            isinstance(self.feature_map[f.feature_id], target_class)]

        if len(feature_requests) == 0:
            logging.info('No {} to calculate'.format(target_class.__name__))
            return

        for request in feature_requests:
            feature_id = request.feature_id
            ranges = self.feature_map[feature_id].storable_ranges(start_dt, end_dt)

            p = Pool(processes=n_jobs, initializer=init_parallel_runner,
                     initargs=(self.__class__, self._extra_feature_classes, self._kwargs), maxtasksperchild=1)

            # Renice the workers
            pool_pids = {pp.pid for pp in p._pool}
            parent = psutil.Process()
            for child in parent.children():
                if child.pid in pool_pids:
                    try:
                        current_niceness = child.nice(0)
                        child.nice(current_niceness - 10)
                    except:
                        pass

            # feature_requests, tickers, start_dt, end_dt = in_args
            input_args = [([request], range[0], range[1]) for range in ranges]

            pbar = tqdm(total=len(input_args))

            def update(*args):
                pbar.update(1)

            for arg in input_args:
                p.apply_async(target_function, args=(arg,), callback=update)

            p.close()
            p.join()

    def _log_feature_id(self, feature_id, feature_params, feature_class):
        conn = self.feature_conn()

        params_to_store = feature_class.storage_params(feature_params, self)

        doc = {
            'feature_id': feature_id,
            'feature': feature_class.__name__,
            'params': params_to_store
        }
        try:
            conn['feature_ids'].insert(doc)
        except DuplicateKeyError:
            pass

    def get_feature_request_id(self, feature_request):
        return feature_request.get_feature_id(self)

    def _instantiate_features(self, feature_requests):
        for feature_request in feature_requests:
            feature_class = self._CLASSES_MAP[feature_request.feature_class_name]
            feature_id = self.get_feature_request_id(feature_request)
            params = feature_request.feature_params(self)
            logging.info("Instantiate {} from params {}".format(feature_id, params))
            if feature_id not in self.feature_map.keys():
                self.feature_map[feature_id] = self._instantiate_feature(feature_request)
                self._log_feature_id(feature_id, params, feature_class)

    def _instantiate_feature(self, feature_request):
        feature_class = self._CLASSES_MAP[feature_request.feature_class_name]

        params = feature_request.feature_params(self)

        return feature_class(self, params, **feature_request.feature_kwargs)

    def _create_feature(self, name, params):
        return self._CLASSES_MAP[name](self, params)

    def delete_all_features_ids(self, feature_requests):
        """
        Delete all the data for all the feature_ids for this feature, not only the data for this feature request
        :param feature_requests:
        """
        self._instantiate_features(feature_requests)
        feature_ids = [request.feature_id for request in feature_requests]
        feature_names = set([request.feature_class_name for request in feature_requests])
        logging.info("Deleting all feature ids for {}".format(feature_names))
        logging.warning("Total deletion starting in 5 seconds...")
        time.sleep(5)
        logging.warning("Starting delete...")

        for feature_id in feature_ids:
            self.feature_map[feature_id].delete_all_feature_ids()

        conn = self.feature_conn()
        for feature_name in feature_names:
            query = {'feature': '{}'.format(feature_name)}
            conn['feature_ids'].delete_many(query)


class FeatureRunner(BaseFeatureRunner):
    def _init_shared_objs(self):

        cassandra_connection = self._kwargs['cassandra_connection'] if 'cassandra_connection' in self._kwargs else \
            get_cassandra_connection()

        fixture_cache = self._kwargs['fixture_cache'] if 'fixture_cache' in self._kwargs else \
            FixtureCache()

        odds_cache = self._kwargs['odds_cache'] if 'odds_cache' in self._kwargs else \
            HistoricalCassandraOddsCache(
                cassandra_connection=cassandra_connection,
                fixture_cache=fixture_cache,
                eager=True)

        odds_cache_parse_false = self._kwargs['odds_cache_parse_false'] if \
            'odds_cache_parse_false' in self._kwargs else \
            HistoricalCassandraOddsCache(
                cassandra_connection=cassandra_connection,
                fixture_cache=fixture_cache,
                eager=True,
                parse=False)

        mysql_client = self._kwargs['mysql_client'] if 'mysql_client' in self._kwargs else \
            MySQLClient.init_from_config(config_name='mysql')

        feature_conn = self._kwargs['feature_conn'] if 'feature_conn' in self._kwargs else \
            MongoPersister.init_from_config('features', auto_connect=True)

        abc_tennis_conn = MongoPersister.init_from_config('abc_tennis_v2', auto_connect=True)
        mysql_client.connect()
        self.SHARED_OBJECTS = {
            'cassandra_connection': cassandra_connection,
            'odds_cache': odds_cache,
            'odds_cache_parse_false': odds_cache_parse_false,
            'fixture_cache': fixture_cache,
            'mysql_client': mysql_client,
            'feature_conn': feature_conn,
            'abc_tennis_conn': abc_tennis_conn,

        }

    def cleanup(self):
        self.SHARED_OBJECTS['cassandra_connection'].shutdown()
        self.SHARED_OBJECTS['mysql_client'].close()

    def get_event_features(self, feature_requests, event_ids, repopulate=False):
        """
        Return a EventFeature by event_id

        Returns dict of event metadata
        :param repopulate: Repopulate parent features
        :param feature_requests:
        :param event_ids:
        :return: {event_id: {feature_name: {}}}
        """
        if repopulate:
            self.delete_features_by_event_ids(feature_requests, event_ids)
        self._instantiate_features(feature_requests)

        for request in feature_requests:
            self.feature_map[request.feature_id].initialize_events(event_ids)

        ret = {}

        for event_id in event_ids:
            event_features = defaultdict(lambda: {})
            for request in feature_requests:
                feature_id = request.feature_id
                feature = self.feature_map[feature_id]
                if not isinstance(feature, EventFeature):
                    raise ValueError("Not a EventFeature")

                logging.info("Requesting {} {}".format(feature_id, event_id))

                feature_result = feature.get_dict_from_event_id(event_id, repopulate=repopulate)

                feature_name = feature.__class__.__name__
                event_features[request.prefix + feature_name].update(feature_result)

            ret[event_id] = dict(event_features)
        return ret

    def get_dataframes_by_event_ids(self, feature_requests, event_ids, repopulate=False,
                                    recompute_missing=True):
        """
        Return a TimeserieFeature by event_ids

        :param feature_requests:
        :param event_ids: list of strings like ['GSM55554', 'GSM55555'] or ['ENP7777', 'ENP7778']
        :return: {event_id: DataFrame} where Dataframe has all the features merged
        """
        self._instantiate_features(feature_requests)
        for request in feature_requests:
            feature_id = request.feature_id
            self.feature_map[feature_id].initialize_events(event_ids)

        ret = {}
        for event_id in event_ids:
            features_df = list()
            for request in feature_requests:
                feature_id = request.feature_id
                feature = self.feature_map[feature_id]

                if not isinstance(feature, TimeSeriesFeature):
                    raise ValueError("Not a TimeSeriesFeature")

                logging.info("Requesting {} {}".format(feature_id, event_id))
                request_df = feature.get_df_from_event_id(event_id, repopulate=repopulate,
                                                          recompute_missing=recompute_missing)

                features_df.append(request_df)
            ret[event_id] = merge_timeseries_features(features_df)
        return ret

    def get_dataframes_by_stickers(self, feature_requests, stickers, repopulate=False):
        """
        Return a TimeserieFeature by sticker

        :param feature_requests:
        :param stickers: with or without bookmakers
        :return: {sticker: DataFrame} where Dataframe has all the features merged
        """
        self._instantiate_features(feature_requests)

        for request in feature_requests:
            feature_id = request.feature_id
            self.feature_map[feature_id].initialize_stickers(stickers)

        ret = {}
        # TODO parallelize
        for sticker in stickers:
            features_df = list()
            for request in feature_requests:
                feature_id = request.feature_id
                feature = self.feature_map[feature_id]

                if not isinstance(feature, TimeSeriesFeature):
                    raise ValueError("Not a TimeSeriesFeature")

                logging.info("Requesting {} {}".format(feature_id, sticker))
                df = feature.get_df_from_sticker(sticker, repopulate=repopulate)
                df.columns = [request.prefix + c for c in df.columns]
                features_df.append(df)
            ret[sticker] = merge_timeseries_features(features_df)
        return ret

    def precompute_event_features(self, feature_requests, events_ids, n_jobs=None, batch_size=50):
        self._precompute(feature_requests, events_ids, n_jobs=n_jobs, batch_size=batch_size,
                         input_type='event_features')

    def precompute_by_event_ids(self, feature_requests, events_ids, n_jobs=None, batch_size=50):
        self._precompute(feature_requests, events_ids, n_jobs=n_jobs, batch_size=batch_size, input_type='event_ids')

    def precompute_by_stickers(self, feature_requests, stickers, n_jobs=None, batch_size=50):
        self._precompute(feature_requests, stickers, n_jobs=n_jobs, batch_size=batch_size, input_type='stickers')

    def get_dataframes_by_stickers_multithread(self, feature_requests, stickers, n_threads=4):
        """
        :param n_threads:
        :param stickers:
        :param feature_requests:
        :return: {sticker: DataFrame} where Dataframe has all the features merged
        """
        self._instantiate_features(feature_requests)

        for request in feature_requests:
            feature_id = request.feature_id
            self.feature_map[feature_id].initialize_stickers(stickers)

        if len(stickers) == 0:
            return {}
        else:
            stickers_lists = [c for c in chunks(stickers, int((len(stickers) + n_threads - 1) / n_threads))]

        threads = list()
        result_queue = Queue()
        for n in range(min(n_threads, len(stickers_lists))):
            t = Thread(target=compute_for_sticker_q,
                       args=(feature_requests, stickers_lists[n], self.feature_map, result_queue))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        ret = dict()
        while not result_queue.empty():
            d = result_queue.get()
            ret.update(d)
        return ret

    def compute_event_features(self, feature_requests, event_ids):
        """
        Compute but do not load into memory.
        """
        self._instantiate_features(feature_requests)

        for request in feature_requests:
            feature_id = request.feature_id
            feature = self.feature_map[feature_id]
            if not isinstance(feature, EventFeature):
                raise ValueError("Not a EventFeature")

            logging.info("Requesting {} for {} events".format(feature_id, len(event_ids)))
            feature.compute_for_event_id(event_ids)
        return

    def compute_dataframes_by_stickers(self, feature_requests, stickers):
        """
        Compute but do not load into memory.
        """
        self._instantiate_features(feature_requests)

        for request in feature_requests:
            feature_id = request.feature_id
            self.feature_map[feature_id].initialize_stickers(stickers)

        for request in feature_requests:
            feature_id = request.feature_id
            feature = self.feature_map[feature_id]
            if not isinstance(feature, TimeSeriesFeature):
                raise ValueError("Not a TimeSeriesFeature")

            logging.info("Requesting {} for {} stickers".format(feature_id, len(stickers)))
            feature.compute_for_stickers(stickers)
        return

    def compute_dataframes_by_event_id(self, feature_requests, event_ids):
        """
        Compute but do not load into memory.
        """
        self._instantiate_features(feature_requests)

        for request in feature_requests:
            feature_id = request.feature_id
            feature = self.feature_map[feature_id]
            if not isinstance(feature, TimeSeriesFeature):
                raise ValueError("Not a TimeSeriesFeature")

            logging.info("Requesting {} for {} events".format(feature_id, len(event_ids)))
            feature.compute_for_event_id(event_ids)
        return

    def delete_features_by_event_ids(self, feature_requests, event_ids):
        """
        Delete the features from the caches
        """
        self._instantiate_features(feature_requests)
        for event_id in event_ids:
            for request in feature_requests:
                feature_id = request.feature_id
                logging.info("Deleting {} {}".format(feature_id, event_id))
                self.feature_map[feature_id].delete_by_event_id(event_id)

    def delete_features_by_stickers(self, feature_requests, stickers):
        """
        Delete the features from the caches
        """
        self._instantiate_features(feature_requests)
        for sticker in stickers:
            for request in feature_requests:
                feature_id = request.feature_id
                logging.info("Deleting {} {}".format(feature_id, sticker))
                self.feature_map[feature_id].delete_by_sticker(sticker)

    def delete_features_every_event_and_sticker(self, feature_requests):
        """
        Delete the feature_ids from the caches
        """
        self._instantiate_features(feature_requests)
        for request in feature_requests:
            feature_id = request.feature_id
            logging.info("Deleting everything for {}".format(feature_id))
            self.feature_map[feature_id].delete_feature_id()




if __name__ == '__main__':
    # Multiprocess example
    runner_ = FeatureRunner()

    stickers_ = [u'BB-EENP2613973-FTPS-1-n7_0', u'BB-EENP2613973-FTPS-1-n7_5', u'BB-EENP2613973-FTPS-1-n8_0',
                 u'BB-EENP2613973-FTPS-1-n8_5',
                 u'BB-EENP2613973-FTPS-1-n9_0', u'BB-EENP2613973-FTPS-1-n9_5', u'BB-EENP2613973-FTPS-2-0_0',
                 u'BB-EENP2613973-FTPS-2-0_5', u'BB-EENP2613973-FTPS-2-10_0', u'BB-EENP2613973-FTPS-2-10_5',
                 u'BB-EENP2613973-FTPS-2-11_0', u'BB-EENP2613973-FTPS-2-11_5', u'BB-EENP2613973-FTPS-2-12_0',
                 u'BB-EENP2613973-FTPS-2-12_5',
                 u'BB-EENP2613973-FTPS-2-13_0', u'BB-EENP2613973-FTPS-2-13_5', u'BB-EENP2613973-FTPS-2-14_0',
                 u'BB-EENP2613973-FTPS-2-14_5', u'BB-EENP2613973-FTPS-2-15_0', u'BB-EENP2613973-FTPS-2-15_5',
                 u'BB-EENP2613973-FTPS-2-16_0', u'BB-EENP2613973-FTPS-2-16_5', u'BB-EENP2613973-FTPS-2-17_0',
                 u'BB-EENP2613973-FTPS-2-17_5', u'BB-EENP2613973-FTPS-2-18_0', u'BB-EENP2613973-FTPS-2-18_5',
                 u'BB-EENP2613973-FTPS-2-19_0', u'BB-EENP2613973-FTPS-2-19_5', u'BB-EENP2613973-FTPS-2-1_0',
                 u'BB-EENP2613973-FTPS-2-1_5', u'BB-EENP2613973-FTPS-2-2_0', u'BB-EENP2613973-FTPS-2-2_5',
                 u'BB-EENP2613973-FTPS-2-3_0', u'BB-EENP2613973-FTPS-2-3_5', u'BB-EENP2613973-FTPS-2-4_0',
                 u'BB-EENP2613973-FTPS-2-4_5', u'BB-EENP2613973-FTPS-2-5_0', u'BB-EENP2613973-FTPS-2-5_5',
                 u'BB-EENP2613973-FTPS-2-6_0', u'BB-EENP2613973-FTPS-2-6_5', u'BB-EENP2613973-FTPS-2-7_0',
                 u'BB-EENP2613973-FTPS-2-7_5', u'BB-EENP2613973-FTPS-2-8_0', u'BB-EENP2613973-FTPS-2-8_5',
                 u'BB-EENP2613973-FTPS-2-9_0', u'BB-EENP2613973-FTPS-2-9_5', u'BB-EENP2613973-FTPS-2-n0_5',
                 u'BB-EENP2613973-FTPS-2-n10_0', u'BB-EENP2613973-FTPS-2-n10_5', u'BB-EENP2613973-FTPS-2-n11_0',
                 u'BB-EENP2613973-FTPS-2-n11_5', u'BB-EENP2613973-FTPS-2-n12_0', u'BB-EENP2613973-FTPS-2-n12_5',
                 u'BB-EENP2613973-FTPS-2-n13_0', u'BB-EENP2613973-FTPS-2-n13_5', u'BB-EENP2613973-FTPS-2-n14_0',
                 u'BB-EENP2613973-FTPS-2-n14_5', u'BB-EENP2613973-FTPS-2-n15_0', u'BB-EENP2613973-FTPS-2-n15_5',
                 u'BB-EENP2613973-FTPS-2-n16_0', u'BB-EENP2613973-FTPS-2-n16_5', u'BB-EENP2613973-FTPS-2-n17_0',
                 u'BB-EENP2613973-FTPS-2-n17_5', u'BB-EENP2613973-FTPS-2-n18_0',
                 u'BB-EENP2613973-FTPS-2-n18_5', u'BB-EENP2613973-FTPS-2-n19_0', u'BB-EENP2613973-FTPS-2-n19_5',
                 u'BB-EENP2613973-FTPS-2-n1_0', u'BB-EENP2613973-FTPS-2-n1_5', u'BB-EENP2613973-FTPS-2-n2_0',
                 u'BB-EENP2613973-FTPS-2-n2_5', u'BB-EENP2613973-FTPS-2-n3_0', u'BB-EENP2613973-FTPS-2-n3_5',
                 u'BB-EENP2613973-FTPS-2-n4_0',
                 u'BB-EENP2613973-FTPS-2-n4_5', u'BB-EENP2613973-FTPS-2-n5_0', u'BB-EENP2613973-FTPS-2-n5_5',
                 u'BB-EENP2613973-FTPS-2-n6_0', u'BB-EENP2613973-FTPS-2-n6_5']

    fr = FeatureRequest('MicroPrice', {'bookmakers': [Bookmakers.PINNACLE_SPORTS, Bookmakers.BETFAIR]}, {})
    # runner.get_dataframes_by_stickers([fr], stickers)
    runner_.precompute_by_stickers([fr], stickers_, n_jobs=6, batch_size=1)
