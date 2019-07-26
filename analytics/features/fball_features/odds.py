from sgmtradingcore.analytics.features.odds import ABCBestOdds
from sgmtradingcore.analytics.features.storage import (
    PickleFileFeatureStorage)
from stratagemdataprocessing.dbutils.mongo import MongoPersister


class FballABCBestOdds(ABCBestOdds):
    def __init__(self, runner, params, **kwargs):
        storage = PickleFileFeatureStorage(self)
        super(FballABCBestOdds, self).__init__(
            runner, params, storage=storage, **kwargs)

    @property
    def default_abc_connection(self):
        return MongoPersister.init_from_config(
            'abc', auto_connect=True)

    @property
    def sport_name(self):
        return 'football'

    @property
    def fixture_meta_col_name(self):
        return 'fixture_meta'
