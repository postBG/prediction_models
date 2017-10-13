from sklearn.ensemble import RandomForestClassifier

from log_data.loader import LogDataLoader
from log_data.processor import split_data, normalize_feature, one_hot_encoder, transform_pay_as_label
from models.sklearn import SklearnHelper
from printers.plt import print_confusion_matrix

if __name__ == '__main__':
    loader = LogDataLoader()
    batch = loader.load_batch(1)
    batch = transform_pay_as_label(batch)
    batch = normalize_feature(batch,
                              ['duration(sec)', 'visit', 'event', 'pv', 'productview', 'cart', 'wishlist', 'order'])
    batch = one_hot_encoder(batch, ['isLogin'])
    train_data, validate_data, test_data = split_data(batch, validation_rate=0)

    rf = SklearnHelper(cls=RandomForestClassifier)

    rf.train(train_data)
    predicted_labels = rf.predict(test_data.drop('pay', axis=1))
    print_confusion_matrix(test_data.pay.apply(lambda pay: 1 if pay > 0 else 0), predicted_labels)
