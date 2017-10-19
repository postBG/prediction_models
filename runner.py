import tensorflow as tf
import numpy as np

from log_data.loader import LogDataLoader
from log_data.processor import FeedForwardProcessor, split_data
from models.neural_network.models import SimpleFeedForwardNetwork
from models.neural_network.utils import mean_false_error, mean_squared_false_error
from metric import get_metrics_using_labels
from printers.plt import print_confusion_matrix

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('date', 10,
                            """테스트에 사용할 날짜""")
tf.app.flags.DEFINE_integer('epochs', 100,
                            """테스트에 사용할 날짜""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size""")
tf.app.flags.DEFINE_float('lr', 0.0001,
                          """Learning Rate""")


def _optimizer(model, lr):
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(model.loss)


def prob_to_labels(probs):
    return np.argmax(probs, axis=1)


def main():
    loader = LogDataLoader()
    batch = loader.load_batch(FLAGS.date)
    data_processor = FeedForwardProcessor()

    batch = data_processor.process(batch)
    train_data, validate_data, test_data = split_data(batch, validation_rate=0.2)

    validate_features, validate_labels = data_processor.separate_features_and_label(validate_data)
    test_features, test_labels = data_processor.separate_features_and_label(test_data)

    inputs = tf.placeholder(tf.float32, shape=[None, 10])
    model = SimpleFeedForwardNetwork(inputs, (128, 128), loss_func=mean_squared_false_error)
    optimizer = _optimizer(model=model, lr=FLAGS.lr)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.epochs):
            for X, y in data_processor.to_mini_batch(train_data, FLAGS.batch_size):
                sess.run(optimizer, feed_dict={
                    model.inputs: X,
                    model.labels: y
                })

            predicted_logits, loss = sess.run([model.logits, model.loss], feed_dict={
                model.inputs: validate_features,
                model.labels: validate_labels
            })

            predicted_labels = prob_to_labels(predicted_logits)
            labels = prob_to_labels(validate_labels.values)
            precision, recall, f1_score, accuracy = get_metrics_using_labels(labels, predicted_labels)
            print("Epoch: {} > Loss: {:2f}, Precision: {:2f}, Recall: {:2f}, F1-Score: {:2f}, Accuracy: {:2f}"
                  .format(epoch, loss, precision, recall, f1_score, accuracy))

        predicted_logits = sess.run(model.logits, feed_dict={
            model.inputs: test_features,
            model.labels: test_labels
        })
        predicted_labels = prob_to_labels(predicted_logits)
        labels = prob_to_labels(test_labels.values)
        print_confusion_matrix(labels, predicted_labels)


if __name__ == '__main__':
    main()
