import tensorflow as tf

from log_data.loader import LogDataLoader
from log_data.processor import FeedForwardProcessor, split_data
from models.neural_network.models import SimpleFeedForwardNetwork

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('date', 10,
                            """테스트에 사용할 날짜""")
tf.app.flags.DEFINE_integer('epochs', 10,
                            """테스트에 사용할 날짜""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size""")
tf.app.flags.DEFINE_float('lr', 0.0001,
                          """Learning Rate""")


def _optimizer(model, lr):
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(model.loss)


def main():
    loader = LogDataLoader()
    batch = loader.load_batch(FLAGS.date)
    data_processor = FeedForwardProcessor()

    batch = data_processor.process(batch)
    train_data, validate_data, test_data = split_data(batch, validation_rate=0)

    minibatch = data_processor.to_mini_batch(train_data, FLAGS.batch_size)

    inputs = tf.placeholder(tf.float32, shape=[None, 10])
    model = SimpleFeedForwardNetwork(inputs, (128, 128))
    optimizer = _optimizer(model=model, lr=FLAGS.lr)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.epochs):
            for X, y in minibatch:
                sess.run(optimizer, feed_dict={
                    model.inputs: X,
                    model.labels: y
                })
            

if __name__ == '__main__':
    main()
