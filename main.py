import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from log_data.loader import LogDataLoader
from log_data.processor import DefaultPreprocessor, ReadyPreProcessor, split_data, separate_features_and_label
from models.sklearn import SklearnHelper
from printers.plt import print_confusion_matrix

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('date', 10,
                            """테스트에 사용할 날짜""")


def main():
    loader = LogDataLoader()
    batch = loader.load_batch(FLAGS.date)
    batch = DefaultPreprocessor().process(batch)
    train_data, validate_data, test_data = split_data(batch, validation_rate=0)
    x_train, y_train = separate_features_and_label(train_data)

    # sm = SMOTE()
    # x_train, y_train = sm.fit_sample(x_train, y_train)
    rf = SklearnHelper(cls=SVC)
    rf.train(x_train, y_train)
    rf.save('models/persisted_models/svm.pkl')

    predicted_labels = rf.predict(test_data.drop('pay', axis=1))
    print_confusion_matrix(test_data.pay.apply(lambda pay: 1 if pay > 0 else 0), predicted_labels)


if __name__ == '__main__':
    main()
