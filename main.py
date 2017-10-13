import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

from log_data.loader import LogDataLoader
from log_data.processor import split_data, normalize_feature, one_hot_encoder, transform_pay_as_label, drop_fields, \
    separate_features_and_label
from models.sklearn import SklearnHelper
from printers.plt import print_confusion_matrix

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('date', 10,
                            """테스트에 사용할 날짜""")

if __name__ == '__main__':
    loader = LogDataLoader()
    batch = loader.load_batch(FLAGS.date)
    batch = transform_pay_as_label(batch)
    batch = normalize_feature(batch,
                              ['duration(sec)', 'visit', 'event', 'pv', 'productview', 'cart', 'wishlist', 'order'])
    batch = one_hot_encoder(batch, ['isLogin'])
    batch = drop_fields(batch, ['pc_id'])
    train_data, validate_data, test_data = split_data(batch, validation_rate=0)

    rf = SklearnHelper(cls=RandomForestClassifier)

    x_train, y_train = separate_features_and_label(train_data)
    rf.train(x_train, y_train)

    predicted_labels = rf.predict(test_data.drop('pay', axis=1))
    print_confusion_matrix(test_data.pay.apply(lambda pay: 1 if pay > 0 else 0), predicted_labels)
