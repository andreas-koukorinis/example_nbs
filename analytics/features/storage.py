import copy
import datetime
import logging
import math
import os
import pickle
import shutil
import warnings

import pymongo
import pytz
from pymongo.bulk import BulkWriteError

from sgmtradingcore.util.misc import chunks
from stratagemdataprocessing.crypto.enums import get_last_valid_datetime
from stratagemdataprocessing.dbutils.mongo import MongoPersister

EXHAUST = pymongo.cursor.CursorType.EXHAUST


class FeatureStorage(object):
    def __init__(self, feature):
        self._feature = feature

    def load_feature_by_event_id(self, feature, event_id):
        """
        Return None if not present
        """
        raise NotImplementedError('Subclass me')

    def store_feature_by_event_id(self, feature_id, event_id, df):
        raise NotImplementedError('Subclass me')

    def delete_feature_by_event_id(self, feature_id, event_id):
        raise NotImplementedError('Subclass me')

    def load_feature_by_sticker(self, feature_id, sticker):
        """
        Return None if not present
        """
        raise NotImplementedError('Subclass me')

    def store_feature_by_sticker(self, feature_id, sticker, df):
        raise NotImplementedError('Subclass me')

    def delete_feature_by_sticker(self, feature_id, sticker):
        raise NotImplementedError('Subclass me')

    def delete_feature_id(self, feature_id):
        """
        Delete this feature_id for all the stickers and all the event_ids
        :param feature_id:
        :return:
        """
        raise NotImplementedError('Subclass me')

    def delete_all_feature_ids(self, feature_name):
        """
        Delete all the feature_ids for this feature
        """
        raise NotImplementedError('Subclass me')

    def missing_stickers(self, feature_id, stickers):
        """
        Return all the missing stickers among the requested ones
        """
        logging.warning("This implementation is too slow! Override it")
        return [sticker for sticker in stickers if
                self.load_feature_by_sticker(feature_id, sticker) is None]

    def missing_events(self, feature_id, event_ids):
        """
        Return all the missing events among the requested ones
        :param feature_id:
        :param event_ids:
        :return:
        """
        logging.warning("This implementation is too slow! Override it")
        return [event_id for event_id in event_ids if
                self.load_feature_by_event_id(feature_id, event_id) is None]


class EmptyFeatureStorage(FeatureStorage):
    """
    Use if you don't want your storage to be backup up by another storage.
    """

    def __init__(self, feature):
        super(EmptyFeatureStorage, self).__init__(feature)

    def load_feature_by_event_id(self, feature_id, event_id):
        return None

    def store_feature_by_event_id(self, feature_id, event_id, df):
        pass

    def delete_feature_by_event_id(self, feature_id, event_id):
        pass

    def load_feature_by_sticker(self, feature_id, sticker):
        return None

    def store_feature_by_sticker(self, feature_id, sticker, df):
        pass

    def delete_feature_by_sticker(self, feature_id, sticker):
        pass

    def load_feature(self, feature_id, start, end):
        return None, None

    def load_previous_feature(self, start_dt):
        return None, (None, None)

    def missing_ranges(self, feature_id, start, end):
        return [(start, end)]

    def storable_ranges(self, start, end):
        return [(start, end)]

    def store_feature(self, df, start, end):
        pass

    def delete_range(self, start, end):
        pass

    def delete_feature_id(self, feature_id):
        pass

    def delete_all_feature_ids(self, feature_name):
        pass

    def missing_tickers(self, feature_id, tickers):
        return tickers

    def missing_stickers(self, feature_id, stickers):
        return stickers

    def missing_events(self, feature_id, event_ids):
        return event_ids


# Deprecated, use MongoCompactTimeseriesStorage instead
class MongoTimeseriesFeatureStorage(FeatureStorage):
    """
    Multiple instances of this cache can exists at the same time, and each of them only cache the features which are
    relevant to it.

    WARNING this is deprecated, use MongoCompactTimeseriesStorage instead
    """

    def __init__(self, feature, mongo_connection=None, mongo_db='features',
                 collection='timeseries'):
        """
        WARNING this is deprecated, use MongoCompactTimeseriesStorage instead
        """
        warnings.warn('MongoTimeseriesFeatureStorage is deprecated,'
                      ' use MongoCompactTimeseriesStorage instead ', DeprecationWarning)
        super(MongoTimeseriesFeatureStorage, self).__init__(feature)
        self._connection = mongo_connection or MongoPersister.init_from_config(mongo_db,
                                                                               auto_connect=True)
        self._collection = collection

    def _get_mongo_docs_by_event_id(self, df, event_id):
        docs = self._feature.get_mongo_docs(df)
        for i in range(len(docs)):
            docs[i].update({'event_id': event_id})
        return docs

    def _get_mongo_docs_by_sticker(self, df, sticker):
        docs = self._feature.get_mongo_docs(df)
        for i in range(len(docs)):
            docs[i].update({'sticker': sticker})
        return docs

    def _get_data_from_mongo_docs_by_event_id(self, docs, event_id):
        """
        :return: DataFrame if self._feature is a TimeSerieFeature, else return a dict()
        """
        df = self._feature.get_df_from_mongo_docs(docs)
        del df['feature_id']
        if 'event_id' in df:
            del df['event_id']
        return df

    def _get_data_from_mongo_docs_by_sticker(self, docs, sticker):
        df = self._feature.get_df_from_mongo_docs(docs)
        del df['feature_id']
        if 'sticker' in df:
            del df['sticker']
        return df

    def load_feature_by_sticker(self, feature_id, sticker):
        query = {'feature_id': feature_id,
                 'sticker': sticker}
        docs = [d for d in self._connection[self._collection].find(query)]
        if len(docs) == 0:
            return None
        return self._get_data_from_mongo_docs_by_sticker(docs, sticker)

    def load_feature_by_event_id(self, feature_id, event_id):
        query = {'feature_id': feature_id,
                 'event_id': event_id}
        docs = [d for d in self._connection[self._collection].find(query)]
        if len(docs) == 0:
            return None
        return self._get_data_from_mongo_docs_by_event_id(docs, event_id)

    def store_feature_by_sticker(self, feature_id, sticker, df):
        docs = self._get_mongo_docs_by_sticker(df, sticker)
        if len(docs):
            res = self._connection[self._collection].insert_many(docs)
            return res

    def store_feature_by_event_id(self, feature_id, event_id, df):
        docs = self._get_mongo_docs_by_event_id(df, event_id)
        if len(docs):
            res = self._connection[self._collection].insert_many(docs)
            return res

    def delete_feature_by_event_id(self, feature_id, event_id):
        query = {'feature_id': feature_id,
                 'event_id': event_id}
        ret = self._connection[self._collection].delete_many(query)
        return ret.deleted_count

    def delete_feature_by_sticker(self, feature_id, sticker):
        query = {'feature_id': feature_id,
                 'sticker': sticker}
        ret = self._connection[self._collection].delete_many(query)
        return ret.deleted_count

    def delete_feature_id(self, feature_id):
        query = {'feature_id': feature_id}
        ret = self._connection[self._collection].delete_many(query)
        return ret.deleted_count


class MongoCompactTimeseriesStorage(FeatureStorage):
    """
    Multiple instances of this cache can exists at the same time, and each of them only cache the features which are
    relevant to it.
    """

    def __init__(self, feature, mongo_connection=None, mongo_db='features',
                 collection_stickers='timeseries_compact_s',
                 collection_events='timeseries_compact_e', max_values=1000):
        from sgmtradingcore.analytics.features.feature import TimeSeriesFeature
        if not isinstance(feature, TimeSeriesFeature):
            raise ValueError("This is optimized for timeseries only")
        super(MongoCompactTimeseriesStorage, self).__init__(feature)
        self._connection = mongo_connection or MongoPersister.init_from_config(mongo_db,
                                                                               auto_connect=True)
        self._collection_e = collection_events
        self._collection_s = collection_stickers
        self._max_values = max_values

    def _get_mongo_docs_by_event_id(self, df, event_id):
        base_doc = {
            'event_id': event_id,
            'feature_id': self._feature.feature_id
        }

        docs = list()
        all_values = self._feature.get_mongo_values(df)
        for i, data in enumerate(chunks(all_values, self._max_values)):
            doc = copy.copy(base_doc)
            doc['chunk'] = i
            doc['series'] = data
            docs.append(doc)

        if len(all_values) == 0:
            doc = copy.copy(base_doc)
            doc['chunk'] = 0
            doc['series'] = []
            docs.append(doc)
        return docs

    def _get_mongo_docs_by_sticker(self, df, sticker):
        base_doc = {
            'sticker': sticker,
            'feature_id': self._feature.feature_id
        }

        docs = list()
        all_values = self._feature.get_mongo_values(df)
        for i, data in enumerate(chunks(all_values, self._max_values)):
            doc = copy.copy(base_doc)
            doc['chunk'] = i
            doc['series'] = data
            docs.append(doc)

        if len(all_values) == 0:
            doc = copy.copy(base_doc)
            doc['chunk'] = 0
            doc['series'] = []
            docs.append(doc)
        return docs

    def _get_data_from_mongo_docs_by_event_id(self, docs, event_id):
        """
        :return: DataFrame if self._feature is a TimeSerieFeature, else return a dict()
        """
        values = list()
        for doc in docs:
            values.extend(doc['series'])
        df = self._feature.get_df_from_mongo_values(values)
        if 'feature_id' in df:
            del df['feature_id']
        if 'sticker' in df:
            del df['sticker']
        return df

    def _get_data_from_mongo_docs_by_sticker(self, docs, sticker):
        values = list()
        for doc in docs:
            values.extend(doc['series'])
        df = self._feature.get_df_from_mongo_values(values)
        if 'feature_id' in df:
            del df['feature_id']
        if 'sticker' in df:
            del df['sticker']
        return df

    def load_feature_by_sticker(self, feature_id, sticker):
        query = {'feature_id': feature_id,
                 'sticker': sticker}
        docs = [d for d in self._connection[self._collection_s].find(query).sort('chunk', 1)]
        if len(docs) == 0:
            return None
        return self._get_data_from_mongo_docs_by_sticker(docs, sticker)

    def load_feature_by_event_id(self, feature_id, event_id):
        query = {'feature_id': feature_id,
                 'event_id': event_id}
        docs = [d for d in self._connection[self._collection_e].find(query).sort('chunk', 1)]
        if len(docs) == 0:
            return None
        return self._get_data_from_mongo_docs_by_event_id(docs, event_id)

    def store_feature_by_sticker(self, feature_id, sticker, df):
        docs = self._get_mongo_docs_by_sticker(df, sticker)
        if len(docs):
            res = self._connection[self._collection_s].insert_many(docs)
            return res

    def store_feature_by_event_id(self, feature_id, event_id, df):
        docs = self._get_mongo_docs_by_event_id(df, event_id)
        if len(docs):
            res = self._connection[self._collection_e].insert_many(docs)
            return res

    def delete_feature_by_event_id(self, feature_id, event_id):
        query = {'feature_id': feature_id,
                 'event_id': event_id}
        ret = self._connection[self._collection_e].delete_many(query)
        return ret.deleted_count

    def delete_feature_by_sticker(self, feature_id, sticker):
        query = {'feature_id': feature_id,
                 'sticker': sticker}
        ret = self._connection[self._collection_s].delete_many(query)
        return ret.deleted_count

    def delete_feature_id(self, feature_id):
        query = {'feature_id': feature_id}
        ret1 = self._connection[self._collection_e].delete_many(query)
        ret2 = self._connection[self._collection_s].delete_many(query)
        return ret1.deleted_count + ret2.deleted_count

    def delete_all_feature_ids(self, feature_name):
        query = {'feature_id': '/^{}_.*/'.format(feature_name)}
        ret1 = self._connection[self._collection_e].delete_many(query)
        ret2 = self._connection[self._collection_s].delete_many(query)
        return ret1.deleted_count + ret2.deleted_count

    def missing_stickers(self, feature_id, stickers):
        stickers = [str(s) for s in stickers]
        present_stickers = set()
        query = {'feature_id': feature_id,
                 'sticker': {'$in': stickers},
                 'chunk': 0,
                 }
        proj = {'sticker': 1}
        docs = [d for d in self._connection[self._collection_s].find(query, proj)]

        for d in docs:
            present_stickers.add(d['sticker'])

        missing = set(stickers) - present_stickers
        return missing

    def missing_events(self, feature_id, event_ids):
        event_ids = [str(s) for s in event_ids]
        present_events = set()
        query = {'feature_id': feature_id,
                 'event_id': {'$in': event_ids},
                 'chunk': 0}
        proj = {'event_id': 1}

        docs = [d for d in self._connection[self._collection_e].find(query, proj)]

        for d in docs:
            present_events.add(d['event_id'])

        missing = set(event_ids) - present_events
        return missing


class MongoEventFeatureStorage(MongoTimeseriesFeatureStorage):
    """ TODO: possibly more efficient to store all events in one doc (per feature.id)
    """

    def __init__(self, feature, mongo_connection=None, mongo_db='features',
                 collection='event_features'):
        from sgmtradingcore.analytics.features.feature import EventFeature
        if not isinstance(feature, EventFeature):
            raise ValueError("wrong feature class: {}".format(feature.__class__.__name__))

        super(MongoEventFeatureStorage, self).__init__(feature,
                                                       mongo_connection=mongo_connection,
                                                       mongo_db=mongo_db,
                                                       collection=collection)

    def _get_data_from_mongo_docs_by_event_id(self, docs, event_id):
        """
        :param docs: [mongo docs]
        :param event_id:
        :return: dict() with all the data for this event. It should not contain 'event_id' or 'feature_id'
        """
        d = self._feature.get_dicts_from_mongo_docs(docs)[event_id]
        del d['feature_id']
        return d

    def _get_mongo_docs_by_sticker(self, df, sticker):
        raise ValueError("EventFeature can only be accessed by event_id")

    def _get_data_from_mongo_docs_by_sticker(self, docs, sticker):
        raise ValueError("EventFeature can only be accessed by event_id")

    def store_feature_by_sticker(self, feature_id, sticker, df):
        raise ValueError("EventFeature can only be accessed by event_id")

    def delete_feature_by_sticker(self, feature_id, sticker):
        raise ValueError("EventFeature can only be accessed by event_id")

    def load_feature_by_sticker(self, feature_id, sticker):
        raise ValueError("EventFeature can only be accessed by event_id")

    def missing_events(self, feature_id, event_ids):
        event_ids = [str(s) for s in event_ids]
        present_events = set()
        query = {'feature_id': feature_id,
                 'event_id': {'$in': event_ids}}
        proj = {'event_id': 1}

        docs = [d for d in self._connection[self._collection].find(query, proj)]

        for d in docs:
            present_events.add(d['event_id'])

        missing = set(event_ids) - present_events
        return missing


class MongoInfiniteTimeseriesStorage(FeatureStorage):
    MIN_DT = pytz.UTC.localize(datetime.datetime(1970, 1, 1, 0, 0))

    def __init__(self, feature, period_per_doc, mongo_connection=None, mongo_db='features',
                 collection_tickers='infinite_timeseries_t', max_values=10000):
        """

        :param period_per_doc: (timedelta) max period for each mongo doc
        :param max_values: max number of values per mongo doc
        """

        from sgmtradingcore.analytics.features.feature import InfiniteTimeSeriesFeature
        if not isinstance(feature, InfiniteTimeSeriesFeature):
            raise ValueError("This for InfiniteTimeSeriesFeature only")

        self._period_per_doc = period_per_doc

        super(MongoInfiniteTimeseriesStorage, self).__init__(feature)
        self._connection = mongo_connection  # or MongoPersister.init_from_config(mongo_db, auto_connect=True)

        self._collection_timeseries_t = collection_tickers
        self._max_values = max_values

        self._last_valid_dt = get_last_valid_datetime()

    @property
    def feature_id(self):
        return self._feature.feature_id

    def _time_to_interval_number(self, dt, is_end=False):
        """
        Starting from self.MIN_DT we enumerate the fixed length storage periods.
        This function returns the bucket that dt belongs to, with a small correction for if you are considering
        the end of a bucket.

        They should be closed at the start of the interval, and open at the end.
        :param dt:
        :param is_end:
        :return:
        """
        # Intervals should be closed at the beginning and open at the end
        n = (dt - self.MIN_DT).total_seconds() / float(self._period_per_doc.total_seconds())
        n_floor = int(math.floor(n))

        if (n_floor == n) and is_end:
            return n_floor - 1

        return n_floor

    def _interval_number_to_time(self, n):
        return n * self._period_per_doc + self.MIN_DT

    def _time_to_interval_start(self, dt, is_end=False):
        """
        This converts the interval number back into a datetime.
        :param dt:
        :param is_end:
        :return:
        """
        return self._time_to_interval_number(dt, is_end=is_end) * self._period_per_doc + self.MIN_DT

    def expand_range_to_storage_intervals(self, start, end):
        """
        This function widens the interval specified by start, end up to the previous and next storage period.
        I.e. if the storage period was 1 hour (vertical bars represent storage intervals

                | --- --- | ------- --- | --- --- | --- --- |
                |         |   start^    |         | end^    |
      expanded_range_start^               expanded_range_end^
        :param start:
        :param end:
        :return:
        """
        interval_start = self._time_to_interval_start(start)
        interval_end = self._time_to_interval_start(end, is_end=True) + \
                       self._period_per_doc - datetime.timedelta(microseconds=1)
        return interval_start, interval_end

    def storable_ranges(self, start, end):
        """
        Returns all storable ranges between start and end
        :param start:
        :param end:
        :return:
        """
        ret = []
        start, end = self.expand_range_to_storage_intervals(start, end)
        while start < end:
            ret.append((start, start + self._period_per_doc - datetime.timedelta(microseconds=1)))
            start += self._period_per_doc
        return ret

    def _df_to_mongo_docs(self, df, start_dt, end_dt):
        """
        do NOT store blocks which are not complete
        TODO: Should this not store if we don't have at least one data point for the next / previous intervals?
        (or is that a mess...)
        """
        first_period_dt = self._time_to_interval_start(start_dt)
        last_period_dt = self._time_to_interval_start(end_dt, is_end=True)

        if first_period_dt > last_period_dt:
            return []

        base_doc = {
            'feature_id': self._feature.feature_id,
            'doc_period_s': int(self._period_per_doc.total_seconds()),
            'col_names': ['timestamp'] + df.columns.tolist()
        }

        first_dt, last_dt = self.expand_range_to_storage_intervals(start_dt, end_dt)

        all_values = self._feature.get_mongo_values(df)

        data_by_doc_timestamp = {}

        dt = first_period_dt
        while dt <= last_period_dt:
            data_by_doc_timestamp[dt] = []
            dt += self._period_per_doc

        for data in all_values:
            t = data[0]
            if not first_dt <= t < last_dt:
                continue
            doc_dt = self._time_to_interval_start(t)
            data_by_doc_timestamp[doc_dt].append(data)

        docs = list()

        dt = first_period_dt
        while dt <= last_period_dt:
            block_data = data_by_doc_timestamp[dt]
            if len(block_data) > 0:
                for i, data in enumerate(chunks(block_data, self._max_values)):
                    doc = copy.copy(base_doc)
                    doc['chunk'] = i
                    doc['series'] = data
                    doc['doc_timestamp'] = dt
                    doc['interval_index'] = self._time_to_interval_number(dt)
                    docs.append(doc)
            else:
                doc = copy.copy(base_doc)
                doc['chunk'] = 0
                doc['series'] = []
                doc['doc_timestamp'] = dt
                doc['interval_index'] = self._time_to_interval_number(dt)
                docs.append(doc)
            dt += self._period_per_doc

        return docs

    def _mongo_docs_to_df(self, docs, start_dt, end_dt):
        values = list()
        for doc in docs:
            values.extend(doc['series'])
        df = self._feature.get_df_from_mongo_values(values, docs[0]['col_names'], start_dt, end_dt)
        if 'feature_id' in df:
            del df['feature_id']
        if 'sticker' in df:
            del df['sticker']
        return df

    def load_previous_tick(self, tick, n=1):
        """
        Loads the previous tick (if it has already been calculated)
        for the given feature_id, ticker
        :param n: the number of ticks to load before this datetime
        :param tick: the datetime of interest
        :return:
        """

        doc_period = int(self._period_per_doc.total_seconds())

        query = {'feature_id': self._feature.feature_id,
                 'doc_period_s': doc_period,
                 'doc_timestamp': {'$lt': tick}
                 }

        docs = self._connection[self._collection_timeseries_t].find(query).sort(
            [('doc_timestamp', pymongo.DESCENDING),
             ('chunk', pymongo.DESCENDING)]).batch_size(20)
        empty_docs = []
        previous_tick = None
        for doc in docs:
            if len(doc['series']) == 0:
                empty_docs.append(doc)
            else:
                df = self._mongo_docs_to_df([doc], pytz.UTC.localize(datetime.datetime.min), tick)
                potential_ticks = df.loc[df.index < tick]
                if len(potential_ticks) == 0:
                    empty_docs.append(doc)
                else:
                    previous_tick = df.loc[df.index < tick].iloc[-1]
                    break

        if previous_tick is None:
            return None

        return previous_tick

    def load_feature(self, feature_id, start_dt, end_dt):
        """
        Load all feature with start_dt <= t < end_dt
        :param feature_id:
        :param start_dt:
        :param end_dt:
        :return:
            Dataframe: data which exists, as a single, maybe incomplete dataframe. If None all are missing
            [(start_dt, end_dt)]: missing ranges
        """
        first_period_start = self._time_to_interval_start(start_dt)
        last_period_start = self._time_to_interval_start(end_dt, is_end=True)

        query = {'feature_id': feature_id,
                 'doc_period_s': int(self._period_per_doc.total_seconds()),
                 '$and': [{'doc_timestamp': {'$gte': first_period_start}},
                          {'doc_timestamp': {'$lte': last_period_start}}]
                 }

        sort_criteria = [('doc_timestamp', pymongo.ASCENDING),
                         ('chunk', pymongo.ASCENDING)]

        docs = list(
            self._connection[self._collection_timeseries_t].find(query, cursor_type=EXHAUST).sort(
                sort_criteria).batch_size(20000000))

        if len(docs) == 0:
            return None, []
        data = self._mongo_docs_to_df(docs, start_dt, end_dt)
        missing = self.missing_ranges(feature_id, start_dt, end_dt)
        return data, missing

    def missing_ranges(self, feature_id, start_dt, end_dt):
        first_period_start = self._time_to_interval_start(start_dt)
        last_period_start = self._time_to_interval_start(end_dt, is_end=True)

        query = {'feature_id': feature_id,
                 'chunk': 0,
                 'doc_period_s': int(self._period_per_doc.total_seconds()),
                 '$and': [{'doc_timestamp': {'$gte': first_period_start}},
                          {'doc_timestamp': {'$lte': last_period_start}}]
                 }

        proj = {'doc_timestamp': 1}
        docs = [d for d in self._connection[self._collection_timeseries_t].find(query, proj).sort(
            [('doc_timestamp', pymongo.ASCENDING)])
                ]
        present_dts = set(d['doc_timestamp'] for d in docs)

        missing = []
        dt = first_period_start
        while dt <= last_period_start:
            end = self._time_to_interval_start(dt + self._period_per_doc)
            if dt not in present_dts:
                missing.append((dt, end))
            dt = end

        return missing

    def store_feature(self, df, start_dt, end_dt):
        if 'ticker' in self._feature.params():
            ticker = self._feature.params()['ticker']
            if end_dt > self._last_valid_dt.get(ticker, pytz.UTC.localize(datetime.datetime.max)):
                logging.warning(
                    "Storing incomplete range: {} {} end_dt={}".format(self._feature.feature_id,
                                                                       ticker, end_dt))

        docs = self._df_to_mongo_docs(df, start_dt, end_dt)
        logging.info("Storing {} docs".format(len(docs)))
        if len(docs):
            try:
                res = self._connection[self._collection_timeseries_t].insert_many(docs)
                return res
            except BulkWriteError as bwe:
                logging.error(
                    "Error storing {} {} {}: {}".format(self._feature.feature_id, start_dt, end_dt,
                                                        bwe.details['writeErrors']))
                raise

    def delete_range(self, start_dt, end_dt):

        first_period_start = self._time_to_interval_start(start_dt)
        last_period_start = self._time_to_interval_start(end_dt, is_end=True)

        query = {'feature_id': self._feature.feature_id,
                 'doc_period_s': int(self._period_per_doc.total_seconds()),
                 '$and': [{'doc_timestamp': {'$gte': first_period_start}},
                          {'doc_timestamp': {'$lte': last_period_start}}]
                 }
        ret = self._connection[self._collection_timeseries_t].delete_many(query)
        return ret.deleted_count

    def delete_feature_id(self, feature_id):
        query = {'feature_id': feature_id}
        ret1 = self._connection[self._collection_timeseries_t].delete_many(query)
        return ret1.deleted_count

    def delete_all_feature_ids(self, feature_name):
        query = {'feature_id': {'$regex': '/^{}_.*/'.format(feature_name)}}

        ret1 = self._connection[self._collection_timeseries_t].delete_many(query)
        return ret1.deleted_count


class InMemoryFeatureStorage(FeatureStorage):
    """
    In memory feature cache.
    Optionally can be backed by another in-file or in-database storage
    This is useful only if you will do multiple request for the same feature with the same params.
    Useless if you are already caching your results, or never asking twice for the same thing.
    """

    def __init__(self, feature, parent_cache=None):
        """
        :param parent_cache: type FeatureStorage, use if you want this cache to be backed by another cache
                             (e.g. a file or db storage)
        """
        super(InMemoryFeatureStorage, self).__init__(feature)
        self._parent_cache = parent_cache or EmptyFeatureStorage(feature)
        self._cache = dict()  # {feature_id: {event_id: data, fixture_id: data}}

    def load_feature_by_event_id(self, feature_id, event_id):
        """
        Return None if not present
        """
        if feature_id in self._cache:
            if event_id in self._cache[feature_id]:
                return self._cache[feature_id][event_id]
        data = self._parent_cache.load_feature_by_event_id(feature_id, event_id)
        if data is not None:
            if feature_id not in self._cache:
                self._cache[feature_id] = dict()
            self._cache[feature_id][event_id] = data
        return data

    def store_feature_by_event_id(self, feature_id, event_id, df):
        if feature_id not in self._cache:
            self._cache[feature_id] = dict()
        self._cache[feature_id][event_id] = df
        self._parent_cache.store_feature_by_event_id(feature_id, event_id, df)

    def delete_feature_by_event_id(self, feature_id, event_id):
        if feature_id in self._cache and event_id in self._cache[feature_id]:
            del self._cache[feature_id][event_id]
        if feature_id in self._cache and len(self._cache[feature_id]) == 0:
            del self._cache[feature_id]
        return self._parent_cache.delete_feature_by_event_id(feature_id, event_id)

    def load_feature_by_sticker(self, feature_id, sticker):
        """
        Return None if not present
        """
        if feature_id in self._cache:
            if sticker in self._cache[feature_id]:
                return self._cache[feature_id][sticker]
        data = self._parent_cache.load_feature_by_sticker(feature_id, sticker)
        if data is not None:
            if feature_id not in self._cache:
                self._cache[feature_id] = dict()
            self._cache[feature_id][sticker] = data
        return data

    def store_feature_by_sticker(self, feature_id, sticker, df):
        if feature_id not in self._cache:
            self._cache[feature_id] = dict()
        self._cache[feature_id][sticker] = df
        self._parent_cache.store_feature_by_sticker(feature_id, sticker, df)

    def delete_feature_by_sticker(self, feature_id, sticker):
        if feature_id in self._cache and sticker in self._cache[feature_id]:
            del self._cache[feature_id][sticker]
        if feature_id in self._cache and len(self._cache[feature_id]) == 0:
            del self._cache[feature_id]
        return self._parent_cache.delete_feature_by_sticker(feature_id, sticker)

    def delete_feature_id(self, feature_id):
        if feature_id in self._cache:
            del self._cache[feature_id]
        return self._parent_cache.delete_feature_id(feature_id)

    def delete_all_feature_ids(self, feature_name):
        for feature_id in self._cache.keys():
            if feature_name == feature_id.split('_')[0]:
                del self._cache[feature_id]
        self._parent_cache.delete_all_feature_ids(feature_name)


class PickleFileFeatureStorage(FeatureStorage):
    """
    Store dataframe in local pickle files. Is not backed by another memory.

    Multiple instances of this cache can exists at the same time, and each of them only cache the features which are
    relevant to it.
    """

    def __init__(self, feature, parent_cache=None, local_cache_dir=None):
        super(PickleFileFeatureStorage, self).__init__(feature)
        self._local_cache_dir = local_cache_dir or os.path.expanduser('~/.feature_cache')
        self._params_folder = os.path.join(self._local_cache_dir, 'params')
        if not os.path.exists(self._local_cache_dir):
            os.makedirs(self._local_cache_dir)
        if not os.path.exists(self._params_folder):
            os.makedirs(self._params_folder)
        self._parent_cache = parent_cache or EmptyFeatureStorage(feature)

    def _get_mongo_docs_by_event_id(self, df, event_id):
        docs = self._feature.get_mongo_docs(df)
        for i in range(len(docs)):
            docs[i].update({'event_id': event_id})
        return docs

    def _get_mongo_docs_by_sticker(self, df, sticker):
        docs = self._feature.get_mongo_docs(df)
        for i in range(len(docs)):
            docs[i].update({'sticker': sticker})
        return docs

    def _get_data_from_mongo_docs_by_event_id(self, docs, event_id):
        """
        :return: DataFrame if self._feature is a TimeSerieFeature, else return a dict()
        """
        df = self._feature.get_df_from_mongo_docs(docs)
        if 'event_id' in df:
            del df['event_id']
        return df

    def _get_feature_id_folder(self, feature_id):
        return os.path.join(self._local_cache_dir, feature_id.split('_')[0], feature_id)

    def _get_file_path_by_event_id(self, feature_id, event_id):
        feature_id_folder = self._get_feature_id_folder(feature_id)
        filename = "{}.pickle".format(event_id)
        return os.path.join(feature_id_folder, filename)

    def _get_file_path_by_sticker(self, feature_id, sticker):
        return self._get_file_path_by_event_id(feature_id, sticker)

    def _get_param_file_path(self, feature_id):
        filename = "{}.txt".format(feature_id)
        return os.path.join(self._params_folder, filename)

    def _write_param_file(self, feature_id):
        filepath = self._get_param_file_path(feature_id)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(str(self._feature.params()))

    def load_feature_by_event_id(self, feature_id, event_id):
        filepath = self._get_file_path_by_event_id(feature_id, event_id)
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r') as f:
            data = pickle.load(f)
        return data

    def store_feature_by_event_id(self, feature_id, event_id, df):
        filepath = self._get_file_path_by_event_id(feature_id, event_id)
        if os.path.exists(filepath):
            logging.warning("File {} already exists".format(filepath))
            return
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(filepath, 'w') as f:
            pickle.dump(df, f)

        self._write_param_file(feature_id)

    def delete_feature_by_event_id(self, feature_id, event_id):
        filepath = self._get_file_path_by_event_id(feature_id, event_id)
        if not os.path.exists(filepath):
            return None
        os.remove(filepath)

    def load_feature_by_sticker(self, feature_id, sticker):
        filepath = self._get_file_path_by_sticker(feature_id, sticker)
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r') as f:
            data = pickle.load(f)
        return data

    def store_feature_by_sticker(self, feature_id, sticker, df):
        filepath = self._get_file_path_by_sticker(feature_id, sticker)
        if os.path.exists(filepath):
            logging.warning("File {} already exists".format(filepath))
            return
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(filepath, 'w') as f:
            pickle.dump(df, f)

        self._write_param_file(feature_id)

    def delete_feature_by_sticker(self, feature_id, sticker):
        filepath = self._get_file_path_by_sticker(feature_id, sticker)
        if not os.path.exists(filepath):
            return None
        os.remove(filepath)

    def delete_feature_id(self, feature_id):
        folderpath = self._get_feature_id_folder(feature_id)
        if not os.path.exists(folderpath):
            return None
        shutil.rmtree(folderpath)
