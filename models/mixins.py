from sklearn.externals import joblib


class ModelPersistMixin:
    def save(self, filename='filename.pkl'):
        joblib.dump(self.clf, filename)
