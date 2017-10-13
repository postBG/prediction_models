from sklearn.ensemble import RandomForestClassifier

from log_data.loader import LogDataLoader
from log_data.processor import split_data, normalize_feature, one_hot_encoder
from models.sklearn import SklearnHelper
from printers.plt import print_confusion_matrix

if __name__ == '__main__':
    loader = LogDataLoader()
    batch = loader.load_batch(1)
    batch = normalize_feature(batch,
                              ['duration(sec)', 'visit', 'event', 'pv', 'productview', 'cart', 'wishlist', 'order'])
    batch = one_hot_encoder(batch, ['isLogin'])
    train_data, validate_data, test_data = split_data(batch)

    rf = SklearnHelper(cls=RandomForestClassifier, params={
        'n_jobs': -1,
        'n_estimators': 500,
        'warm_start': True,
        # 'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'verbose': 0
    })

    rf.train(train_data)
    predicted_labels = rf.predict(validate_data.drop('pay', axis=1))
    print_confusion_matrix(validate_data.pay.apply(lambda pay: 1 if pay > 0 else 0), predicted_labels)
