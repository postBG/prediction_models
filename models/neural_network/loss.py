import tensorflow as tf


def cross_entropy(labels, logits):
    return tf.reduce_mean(tf.losses.softmax_cross_entropy(labels, logits))


def mean_false_error(labels, logits):
    """

    :param labels: tensor of int32
    :param logits: tensor of float32
    :return:
    """
    output = tf.nn.softmax(logits)
    fpe = _false_positive_error(labels, output)
    fne = _false_negative_error(labels, output)

    return tf.add(fpe, fne)


def mean_squared_false_error(labels, logits):
    output = tf.nn.softmax(logits)
    return _mean_squared_false_error(labels, output)


def _mean_squared_false_error(labels, output):
    fpe = _false_positive_error(labels, output)
    fne = _false_negative_error(labels, output)

    return tf.add(20 * tf.square(fpe), tf.square(fne))


def _false_positive_error(labels, output):
    negative_labels, negative_output, negative_num = filter_labels([1, 0], labels, output)
    squared_error = tf.reduce_sum(tf.square(tf.subtract(negative_labels, negative_output)))
    return tf.divide(squared_error, 2 * tf.cast(negative_num, tf.float32))


def _false_negative_error(labels, output):
    positive_labels, positive_output, positive_num = filter_labels([0, 1], labels, output)
    squared_error = tf.reduce_sum(tf.square(tf.subtract(positive_labels, positive_output)))
    return tf.divide(squared_error, 2 * tf.cast(positive_num, tf.float32))


def filter_labels(target_label, labels, output):
    """

    :param target_label:
        tensor or tensor-like int32
        negative: [1, 0], positive: [0, 1]
    :param labels:
        tensor int32
    :param output:
        tensor float32
    :return: (filtered_label, filtered_output, num_of_elements)
        negative인 경우, negative인 row만 값을 살리고 나머지는 0으로 채움.
        positive인 경우도 마찬가지
    """
    indicators = tf.cast(tf.equal(labels, target_label), tf.float32)
    return tf.multiply(indicators, tf.cast(labels, tf.float32)), tf.multiply(indicators, output), tf.div(
        tf.count_nonzero(indicators), 2)
