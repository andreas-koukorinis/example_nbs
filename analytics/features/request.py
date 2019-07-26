class FeatureRequest(object):
    def __init__(self, feature_class_name, feature_params=None, feature_kwargs=None, prefix='', config=None):
        """

        :param feature_class_name:
        :param feature_params: type dict, must be jsonable
        :param feature_kwargs:
        :param prefix: add a prefix to column names or dict keys.
        """
        self.feature_class_name = feature_class_name
        # json.dumps(feature_params)
        self._feature_params = feature_params or {}
        self.feature_kwargs = feature_kwargs or {}
        self.feature_config = config
        self._feature_id = None
        self.prefix = prefix

    @classmethod
    def from_feature(cls, feature_obj):
        feature_class_name = feature_obj.__class__.__name__
        feature_params = feature_obj.params()
        return cls(feature_class_name, feature_params)

    @property
    def feature_id(self):
        if self._feature_id is None:
            raise ValueError("Not computed yet")
        return self._feature_id

    def get_feature_id(self, runner):
        if self._feature_id is None:
            c = self.get_feature_class(runner)
            self._feature_id = c.get_feature_id(c.__name__, self.feature_params(runner),
                                                runner, feature_conn=runner.shared_objects().get('feature_conn'))
        return self._feature_id

    def get_feature_class(self, runner):
        return runner.classes_map()[self.feature_class_name]

    def feature_params(self, runner):
        c = self.get_feature_class(runner)
        return c.sanitize_parameters(c.apply_config(self.feature_config, self._feature_params), runner)

