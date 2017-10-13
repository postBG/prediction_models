from log_data.processor import separate_features_and_label, drop_fields


class SklearnHelper:
    """
    사용방법:
        rf = SklearnHelper(clf=RandomForestClassifier, params=rf_params)
    """

    def __init__(self, cls, params=None):
        params = params if params else {}
        self.clf = cls(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def feature_importances(self, x, y):
        print(self.train(x, y).feature_importances_)
