import copy
import datetime
import json
import logging
from datetime import timedelta

import pandas as pd
import pytz
from pandas.core.frame import DataFrame

from sgmtradingcore.analytics.features.request import FeatureRequest
from sgmtradingcore.analytics.features.storage import EmptyFeatureStorage
from stratagemdataprocessing.crypto.enums import get_last_valid_datetime, get_first_valid_datetime
from stratagemdataprocessing.dbutils.mongo import get_query_from_document, MongoPersister
from stratagemdataprocessing.util.hashing import deterministic_hash


def append_timeseries_features(dfs):
    """
    Merge
    :param dfs: [DataFrame]
    :return:
    """
    dfs = [df for df in dfs if not df.empty]
    if len(dfs) == 0:
        return TimeSeriesFeature.get_empty_feature()

    if len(dfs) == 1:
        return dfs[0]

    df = dfs[0].copy()

    for d in dfs[1:]:
        if not d.empty:
            df = df.append(d, verify_integrity=True)

    df.sort_index(inplace=True)
    return df


def merge_timeseries_features(dfs):
    """
    Merge
    :param dfs: [DataFrame]
    :return:
    """
    if len(dfs) == 0:
        return TimeSeriesFeature.get_empty_feature()
    if len(dfs) == 1:
        return dfs[0]

    df = dfs[0].copy()
    for d in dfs[1:]:
        if not d.empty:
            df = pd.merge(df, d, left_index=True, right_index=True, how='outer', sort=True)

    return df


class MissingParameterError(Exception):
    pass


class Feature(object):
    """
    Feature class define how to compute and store in memory the feature.
    Each feature has a feature_id which depends from the class and the params used.
    One feature instance exists for each feature_id

    REQUIRED_PARAMETERS is a set of strings defining which parameters are considered by the features
    """

    INDEX_FIELD = None
    REQUIRED_PARAMETERS = set()
    RECURSIVE = False

    def __init__(self, runner, params, storage, **kwargs):
        """
        Feature_id depends from the calss name and params, but not from storage or kwargs.
        Only initialise features using a runner
        :param runner: FeatureRunner()
        :param params: params which define the feature_id. must be a dict(). Order counts in lists.
        :param storage: FeatureStorage()
        :param kwargs: params which are not part of feature_id. Usually empty.
        """
        self._params = self.sanitize_parameters(params, runner)
        self._feature_id = None
        self._runner = runner
        self._storage = storage

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - '
                                   '%(levelname)s - %(message)s')

    @property
    def runner(self):
        return self._runner

    @classmethod
    def _check_parameters(cls, params):
        """
        A final check on the parameters that is called post init (for type checking).
        If there is a problem raise an error
        :return: None
        """
        pass

    @classmethod
    def check_parameters(cls, params):
        """
        Wrapper to call all _check_parameters in parent classes
        :return: None
        """
        for superclass in cls.mro():
            if issubclass(superclass, Feature):
                superclass._check_parameters(params)

    def params(self):
        return self._params

    @classmethod
    def required_parameters(cls):
        p = set()

        for superclass in cls.mro():
            if issubclass(superclass, Feature):
                p = p.union(superclass.REQUIRED_PARAMETERS)
        return p

    @classmethod
    def default_parameters(cls, params, runner):
        """
        Default parameters are built up over the class hierarchy.
        The aim of this is to fill in any parameters that aren't specified explicity with sensible defaults.
        :param params:
        :param runner:
        :return:
        """


        base_params = {}
        for superclass in reversed(cls.mro()):
            if issubclass(superclass, Feature):
                current_required = superclass.required_parameters()
                subset_params = {k: v for k, v in params.iteritems() if k in current_required}
                defaults = superclass._default_parameters(subset_params, runner)
                defaults.update({k: v for k, v in params.iteritems() if k in defaults})
                base_params.update({k: v for k, v in defaults.iteritems() if k in current_required})
        return base_params

    @classmethod
    def get_config(cls, config_name):
        """
        get_config plays a complimentary role to default_parameters().

        The config is only taken from the class that is instantiated (i.e. it is not built up over the class hierarchy
        like default_parameters(). This allows a shortcut in the FeatureRequest if there are frequently used
        configurations for a feature.

        The order of priority for params is
        1. FeatureRequest parameters
        2. Configuration
        3. default_parameters()

        i.e. any params explicitly specified in the FeatureRequest will override anything in a configuration which,
        in turn, will override anything in the default_parameters().

        Note: configs should only be used in class methods. An instance of a class has no idea what config it was
        produced from and only sees the final result of generating the parameters in the process above. This ensures
        that features are idempotent with respect to the input parameters only. Anything that breaks this will violate
        the assumptions around the hashing and caching procedure.

        :param config_name:
        :return: dict of params
        """
        if config_name is None:
            return {}
        else:
            raise ValueError('Unknown config')

    @classmethod
    def _default_parameters(cls, params, runner):
        """
        These are the defaults for any parameters that aren't passed in.

        The defaults /can/ be a function of the parameters passed in, i.e. if you choose
        one bookmaker you may be interested in e.g. a higher frequency aggregation typically.
        However, if you can avoid doing this then you probably should.

        You will only see params in this feature's required_parameters() in this function
        :param runner:
        :param params:
        :return: dict of params
        """
        return {}

    @classmethod
    def sanitize_parameters(cls, params, runner):
        """
        If default parameters is overloaded in the subclass but a list of permitted params is not present we fill in
        missing fields

        If permitted params is present then we restrict the input params to those that are permitted
        NB: permitted params from all superclasses are joined together

        Does not check types, just presence. See _check_parameters for checking type
        :param runner:
        :param params:
        :return:
        """
        all_parameters = cls.required_parameters()
        default_parameters = cls.default_parameters(params, runner)

        out_params = {}

        for k in all_parameters:
            if k not in params and k not in default_parameters:
                raise MissingParameterError(
                    '{} requires the parameter {}, which isn''t set'.format(cls.__name__, k))

            if k not in params:
                out_params[k] = default_parameters[k]
            else:
                out_params[k] = params[k]

        cls.check_parameters(out_params)
        return out_params

    @property
    def feature_id(self):

        params = self.sanitize_parameters(self._params, self._runner)

        if self._feature_id is None:
            self._feature_id = self.get_feature_id(self.__class__.__name__, params, self._runner)
        return self._feature_id

    @staticmethod
    def serialize_for_hashing(runner):
        def handle_feature_requests(d):
            if isinstance(d, FeatureRequest):
                return runner.get_feature_request_id(d)
            elif isinstance(d, datetime.datetime):
                return d.isoformat()
            elif isinstance(d, datetime.timedelta):
                return '__timedelta__ {}'.format(d.total_seconds())
            else:
                raise TypeError

        return handle_feature_requests

    @staticmethod
    def get_query_from_document(params, prefix='', runner=None):
        return get_query_from_document(params, prefix=prefix, default=Feature.serialize_for_hashing(runner))

    @classmethod
    def storage_params(cls, params, runner):
        params = cls.sanitize_parameters(params, runner)
        return json.loads(json.dumps(params, default=Feature.serialize_for_hashing(runner)))

    @classmethod
    def apply_config(cls, config_name, params):
        config_params = cls.get_config(config_name)
        config_params.update(params)
        return config_params

    @classmethod
    def get_feature_id(cls, feature_class_name, params, runner, feature_conn=None, force_new=False):

        params = cls.sanitize_parameters(params, runner)

        feature_id = None
        if not force_new:
            feature_id = cls._get_existing_feature_id(feature_class_name, params,
                                                      feature_conn=feature_conn, runner=runner)
        if feature_id is not None:
            return feature_id
        params_query = cls.get_query_from_document(params, runner=runner)
        return feature_class_name + '_' + deterministic_hash(params_query)

    @classmethod
    def _get_existing_feature_id(cls, feature_name, params,
                                 feature_conn=None, runner=None):
        """Check the features database to see if the id for the params exist

        It transforms the document into a query and makes sure that a query
        from the matching document would have had exactly the same query

        Parameters
        --------
        params: dict, parameters of the feature
        feature_name: str, name of the feature
        feature_conn: open connection to the features database

        Returns
        -------
        feature_id:
            - str, if the feature_id found associated with params is found
            - None otherwise (parameters not found in the database)

        """
        close = False
        params = cls.sanitize_parameters(params, runner)

        if feature_conn is None:
            feature_conn = runner.feature_conn() or MongoPersister.init_from_config('features',
                                                                                    auto_connect=True)
            close = True

        query = cls.get_query_from_document(params, prefix='params', runner=runner)
        query['feature'] = feature_name

        existing_params = feature_conn.db['feature_ids'].find(query)
        existing_params.sort('feature_id', 1)
        feature_id = None
        for p in existing_params:
            pp = {'params': p['params'],
                  'feature': p['feature']}
            new_params_check_query = get_query_from_document(pp)
            if new_params_check_query == query:
                feature_id = p['feature_id']
                break
        if close:
            feature_conn.close()
        return feature_id

    @property
    def storage(self):
        return self._storage

    def get_df_from_event_id(self, event_id, repopulate=False, recompute_missing=True):
        """
        Get a dataframe with one or more columns, where the column name typically is NOT the event_id e.g. could be
        one column per sticker or call the column as the feature name
        :param recompute_missing: Recompute anything that is empty
        :param event_id: type str, like 'ENP222222' or 'GSM33333'
        :param repopulate: repopulate the feature for this event_id but not its dependencies
        :return:
        """
        if repopulate:
            self._storage.delete_feature_by_event_id(self.feature_id, event_id)
            df = None
        else:
            df = self._storage.load_feature_by_event_id(self.feature_id, event_id)

        if df is None and recompute_missing:
            df = self._compute_by_event_id(event_id, repopulate=repopulate)
            if df is not None:
                self._storage.store_feature_by_event_id(self.feature_id, event_id, df)
            else:
                df = self.get_empty_feature()
        return df

    def get_df_from_sticker(self, sticker, repopulate=False):
        """
        :param sticker: Note that stickers with and without the bookmaker are considered different
        :param repopulate: repopulate the feature for this event_id but not its dependencies
        """
        if repopulate:
            self._storage.delete_feature_by_sticker(self.feature_id, sticker)

        df = self._storage.load_feature_by_sticker(self.feature_id, sticker)

        if df is None:
            df = self._compute_by_sticker(sticker)
            if df is not None:
                self._storage.store_feature_by_sticker(self.feature_id, sticker, df)
            else:
                df = self.get_empty_feature()

        return df

    def _compute_by_event_id(self, event_id, repopulate=False):
        """
        Compute the feature to be stored in cache.
        Override this method.

        You have access to the FeatureRunner from here
        :param repopulate: if True you might need to repopulate the features you depend from
        :return features for the event_id, the type depends on the implementation of the subclass.
              if empty and should be stored in storage, return self.get_empty_feature()
              if empty and should not be stored in storage because it should be recomputed later, return None.
        """

        raise NotImplementedError()

    def _compute_by_sticker(self, sticker):
        """
        Compute the feature to be stored in cache.
        Override this method.

        You have access to the FeatureRunner from here
        :return features for the sticker, the type depends on the implementation of the subclass.
              if empty and should be stored in storage, return self.get_empty_feature()
              if empty and should not be stored in storage because it should be recomputed later, return None.
        """

        raise NotImplementedError()

    def compute_for_stickers(self, stickers, repopulate=False):
        missing_stickers = self._storage.missing_stickers(self.feature_id, stickers)
        logging.info("Missing {} stickers".format(len(missing_stickers)))
        for sticker in missing_stickers:
            self.get_df_from_sticker(sticker, repopulate=repopulate)
        return

    def compute_for_event_id(self, event_ids, repopulate=False):
        missing_events = self._storage.missing_events(self.feature_id, event_ids)
        logging.info("Missing {} events".format(len(missing_events)))
        for event_id in missing_events:
            self.get_df_from_event_id(event_id, repopulate=repopulate)
        return

    def delete_by_event_id(self, event_id):
        self._storage.delete_feature_by_event_id(self.feature_id, event_id)

    def delete_by_sticker(self, sticker):
        self._storage.delete_feature_by_sticker(self.feature_id, sticker)

    def delete_feature_id(self):
        self._storage.delete_feature_id(self.feature_id)

    def delete_all_feature_ids(self):
        self._storage.delete_all_feature_ids(self.__class__.__name__)

    def get_mongo_docs(self, df):
        """
        Return mongo documents
        Override if you need
        :param df: same type as returned by _compute_by_sticker() or _compute_by_event_id()
        """
        docs = []
        for index, row in df.iterrows():
            doc = dict()
            doc[self.INDEX_FIELD] = index
            doc['feature_id'] = self.feature_id
            doc.update(row.to_dict())
            docs.append(doc)
        return docs

    def get_df_from_mongo_docs(self, mongo_docs):
        """
        Transform mongo docs into the proper format
        Override if you need

        :return : same type as returned by _compute_by_sticker() or _compute_by_event_id()
        """
        timestamps = []
        docs = []
        for d in mongo_docs:
            doc = copy.copy(d)
            timestamps.append(doc[self.INDEX_FIELD])
            del doc[self.INDEX_FIELD]
            del doc['_id']
            docs.append(doc)
        df = pd.DataFrame(docs, index=timestamps)
        return df

    def get_mongo_values(self, df):
        """
        Override if you need
        """
        values = []
        for index, row in df.iterrows():
            doc = dict()
            doc[self.INDEX_FIELD] = index.to_pydatetime()
            doc.update(row.to_dict())
            values.append(doc)
        return values

    def get_df_from_mongo_values(self, values, col_names=None, start_dt=None, end_dt=None):
        """
        Override if you need
        """
        timestamps = []
        lines = []
        for d in values:
            line = copy.copy(d)
            timestamps.append(line[self.INDEX_FIELD])
            del line[self.INDEX_FIELD]
            lines.append(line)
        df = pd.DataFrame(lines, index=timestamps)
        df.index.name = self.INDEX_FIELD
        return df

    def initialize_events(self, event_ids):
        """
        Called once when performing each group of feature requests. I.e. may be useful to do one large query for all
        event_ids and then split up later when calculating the feature on an event by event basis
        :param event_ids: [str]
        """
        pass

    def initialize_stickers(self, stickers):
        """
        Perform some initialization if needed before producing dataframes for these stickers
        :param stickers: with or without bookmakers
        """
        pass

    def initialize(self, start_dt, end_dt):
        """
        Perform some initialization if needed before producing dataframes for these crypto tickers
        :param start_dt:
        :param end_dt:

        """
        pass

    @classmethod
    def get_empty_feature(cls):
        """
        What should be return when the feature is not found
        """
        raise NotImplementedError()


class TimeSeriesFeature(Feature):
    INDEX_FIELD = 'timestamp'

    def __init__(self, runner, params, storage=None, **kwargs):
        """
        Handle timeseries presented as DataFrame indexed by datatime
        """
        storage = storage or EmptyFeatureStorage(self)
        super(TimeSeriesFeature, self).__init__(runner, params, storage)

    def _compute_by_event_id(self, event_id, repopulate=False):
        """
        Compute the feature to be stored in cache.
        Override this method.

        You have access to the FeatureRunner from here
        :param repopulate: if True you might need to repopulate the features you depend from
        :return  DataFrame, where columns are features of the event, and index timestamp values
              if empty and should be stored in storage, return self.get_empty_feature()
              if empty and should not be stored in storage because it should be recomputed later, return None.

        Examples
        --------
        >>> out = DataFrame(
        >>>     data={'self.__class__.__name__': [0, 1]},
        >>>     index=[datetime.datetime(2017, 12, 20, 0, 0, 1, tzinfo=pytz.utc),
        >>>            datetime.datetime(2017, 12, 20, 0, 0, 2, tzinfo=pytz.utc)])
        >>> return out

        """
        raise NotImplementedError()

    @classmethod
    def get_empty_feature(cls):
        """
        What should be return when the feature is not found

        Returns
        -------
        empty DataFrame with the index named correctly
        """
        index = pd.Index([], name=cls.INDEX_FIELD)
        return pd.DataFrame(index=index)

    def _compute_by_sticker(self, sticker, repopulate=False):
        """
        Compute the feature to be stored in cache.
        Override this method.

        You have access to the FeatureRunner from here
        :param repopulate: if True you might need to repopulate the features you depend from
        :return  DataFrame, where columns are features of the sticker, and index timestamp values
              if empty and should be stored in storage, return self.get_empty_feature()
              if empty and should not be stored in storage because it should be recomputed later, return None.

        Examples
        --------
        >>> out = DataFrame(
        >>>     data={'self.__class__.__name__': [0, 1]},
        >>>     index=[datetime.datetime(2017, 12, 20, 0, 0, 1, tzinfo=pytz.utc),
        >>>            datetime.datetime(2017, 12, 20, 0, 0, 2, tzinfo=pytz.utc)])
        >>> return out

        """
        raise NotImplementedError()


class InfiniteTimeSeriesFeature(Feature):
    """
    For timeseries which have no end
    """

    def _compute_by_event_id(self, fixture_id, repopulate=False):
        raise ValueError('No event ids for {}'.format(self.__class__.__name__))

    def _compute_by_sticker(self, sticker):
        raise ValueError('No stickers for {}'.format(self.__class__.__name__))

    INDEX_FIELD = 'timestamp'

    def __init__(self, runner, params, storage=None, storage_period=None, **kwargs):
        """
        Handle timeseries presented as DataFrame indexed by datatime
        """
        storage = storage or EmptyFeatureStorage(self)
        super(InfiniteTimeSeriesFeature, self).__init__(runner, params, storage)
        self._storage_period = storage_period or timedelta(minutes=0)

        self._first_valid_datetimes = get_first_valid_datetime()
        self._last_valid_datetimes = get_last_valid_datetime()

    def storable_ranges(self, start, end):
        return self._storage.storable_ranges(start, end)

    def _check_ticker_range(self, start_dt, end_dt):

        ticker = self._params.get("ticker")
        if ticker is None:
            return

        if ticker not in self._first_valid_datetimes or ticker not in self._last_valid_datetimes:
            return

        first_valid_dt = self._first_valid_datetimes[ticker]

        if start_dt.date() < first_valid_dt.date() or end_dt.date() > self._last_valid_datetimes[ticker].date():
            self._logger.warn("{}: Range {}, {} extends beyond valid dates {}, {} for ticker: {}.".format(
                self.__class__.__name__,
                start_dt,
                end_dt,
                self._first_valid_datetimes[ticker],
                self._last_valid_datetimes[ticker],
                ticker))

    def get_df(self, start_dt, end_dt, repopulate=False):

        if repopulate:
            self._storage.delete_range(start_dt, end_dt)

        df, missing_ranges = self._storage.load_feature(self.feature_id, start_dt, end_dt)
        dfs = [df] if df is not None else []

        if df is None:
            missing_ranges = self._storage.storable_ranges(start_dt, end_dt)

        for missing in missing_ranges:
            logging.info("computing and storing missing range: {}".format(missing))
            self._check_ticker_range(missing[0], missing[1])
            df = self._compute(missing[0], missing[1])
            df = self._check_computation(df, missing[0], missing[1])
            if df is not None:
                self._store_feature(df, missing[0], missing[1])
                dfs.append(df)

        df = append_timeseries_features(dfs)
        df = df.loc[start_dt: end_dt - timedelta(microseconds=1)]
        return df

    def _store_feature(self, df, start, end):
        self._storage.store_feature(df, start, end)

    def delete_range(self, start_dt, end_dt):
        self._storage.delete_range(start_dt, end_dt)

    def _compute(self, start_dt, end_dt):
        raise NotImplementedError()

    def compute(self, start_dt, end_dt, repopulate=False):
        if repopulate:
            self._storage.delete_range(start_dt, end_dt)

        missing_ranges = self._storage.missing_ranges(self.feature_id, start_dt, end_dt)
        for missing in missing_ranges:
            df = self._compute(missing[0], missing[1])
            df = self._check_computation(df, missing[0], missing[1])
            if df is not None:
                self._storage.store_feature(df, missing[0], missing[1])

        return None

    def _check_computation(self, df, start, end):
        # To be overridden in subclasses that are also superclasses (to avoid duplication of code at the bottom level)
        if df is not None and len(df) > 0:
            if df.index.tzinfo != pytz.UTC:
                raise ValueError('Should have timezone in index')
        return df

    def get_mongo_values(self, df):
        """
        Override if you need
        """
        values = []
        for index, row in df.iterrows():
            doc = list()
            doc.append(index.to_pydatetime())
            doc.extend(row.tolist())
            values.append(doc)
        return values

    def get_df_from_mongo_values(self, values, col_names=None, start_dt=None, end_dt=None):

        # col_names
        timestamps = []
        columns = {n: [] for n in col_names}

        for line in values:
            t = line[0].replace(tzinfo=pytz.UTC)
            if not start_dt <= t < end_dt:
                continue

            for i, datum in enumerate(line):
                columns[col_names[i]].append(datum)

            timestamps.append(t)
        del columns[self.INDEX_FIELD]
        df = pd.DataFrame(columns, index=timestamps)
        df.index.name = self.INDEX_FIELD
        df = df[col_names[1:]]
        return df

    @classmethod
    def get_empty_feature(cls):
        """`
        What should be return when the feature is not found

        Returns
        -------
        empty DataFrame with the index named correctly
        """
        index = pd.Index([], name=cls.INDEX_FIELD)
        return pd.DataFrame(index=index)


class EventFeature(Feature):
    INDEX_FIELD = 'event_id'

    def __init__(self, runner, params, storage=None, **kwargs):
        """
        Note: this returns dicts, not dataframes
        :param runner:
        :param params:
        :param storage:
        :param kwargs:
        """
        storage = storage or EmptyFeatureStorage(self)
        super(EventFeature, self).__init__(runner, params, storage)

    def _compute_by_event_id(self, event_id, repopulate=False):
        """
        :param repopulate: if True you might need to repopulate the features you depend from
        :return dict, with features name as keys and their values as values
            if empty and should be stored in storage, return self.get_empty_feature
            if empty and should not be stored in storage because it should be
            recomputed later, return None
        """
        raise NotImplementedError()

    def get_dict_from_event_id(self, event_id, repopulate=False):
        """
        Should return dict
        :param event_id:
        :param repopulate:
        :return:
        """
        if repopulate:
            d = None
        else:
            d = self._storage.load_feature_by_event_id(self.feature_id, event_id)

        if d is None:
            d = self._compute_by_event_id(event_id, repopulate=repopulate)
            if d is not None:
                self._storage.store_feature_by_event_id(self.feature_id, event_id, d)
            else:
                d = self.get_empty_feature()

        return d

    def get_mongo_docs(self, df):
        """
        Override if you need
        :param df: type dict
        """
        doc = {'feature_id': self.feature_id,
               'values': [{'k': k, 'v': v} for k, v in df.iteritems()]}
        return [doc]

    def get_dicts_from_mongo_docs(self, mongo_docs):
        """
        Override if you need
        :return {event_id: dict()}
        """
        docs = {}
        for d in mongo_docs:
            doc = copy.copy(d)
            del doc['_id']
            for val in doc['values']:
                doc[val['k']] = val['v']
            del doc['values']
            docs[doc[self.INDEX_FIELD]] = doc
            del doc[self.INDEX_FIELD]
        return docs

    def get_df_from_sticker(self, sticker, repopulate=False):
        raise ValueError('EventFeature can only by called by event_id')

    def _compute_by_sticker(self, sticker):
        raise ValueError('EventFeature can only by called by event_id')

    def delete_by_sticker(self, sticker):
        raise ValueError('EventFeature can only by called by event_id')

    def initialize_events(self, event_ids):
        """
        Called once when performing each group of feature requests. I.e. may be useful to do one large query for all
        event_ids and then split up later when calculating the feature on an event by event basis
        :param event_ids: [str]
        """
        pass

    @classmethod
    def get_empty_feature(cls):
        """
        What should be return when the feature is not found

        Returns
        -------
        empty dict
        """
        return {}
