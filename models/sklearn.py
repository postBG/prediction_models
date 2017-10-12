def separate_features_and_label(data):
    y = data['pay']
    X = data.drop('pay', axis=1)

    return X, y


def remove_pc_ids(features):
    return features.drop('pc_id', axis=1)


class SklearnHelper:
    """
    사용방법:
        rf = SklearnHelper(clf=RandomForestClassifier, params=rf_params)
    """

    def __init__(self, cls, params=None):
        self.clf = cls(**params)

    def train(self, train_data):
        x_train, y_train = separate_features_and_label(train_data)
        x_train = remove_pc_ids(x_train)
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        x = remove_pc_ids(x)
        return self.clf.predict(x)

    def feature_importances(self, train_data):
        print(self.train(train_data).feature_importances_)
