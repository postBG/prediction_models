import tensorflow as tf

from log_data.loader import LogDataLoader
from log_data.processor import FeedForwardProcessor, split_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('date', 10,
                            """테스트에 사용할 날짜""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size""")


def main():
    loader = LogDataLoader()
    batch = loader.load_batch(FLAGS.date)
    data_processor = FeedForwardProcessor()

    batch = data_processor.process(batch)
    train_data, validate_data, test_data = split_data(batch, validation_rate=0)

    minibatch = data_processor.to_mini_batch(train_data, FLAGS.batch_size)


if __name__ == '__main__':
    main()
