import tensorflow as tf

from models.neural_network.layers import fully_connected, dropout


def _get_hidden_layer_name(idx):
    return 'hidden_layer' + str(idx)


class SimpleFeedForwardNetwork:
    def __init__(self, inputs, layer_shape):
        self.inputs = inputs
        self.labels = tf.placeholder(tf.int32, name='labels')

        self.layer_shape = layer_shape
        self.layers = {
            'inputs': self.inputs
        }

        self.build_graph()

    def build_graph(self):
        layer = self.inputs
        for idx, n_units in enumerate(self.layer_shape):
            hidden_layer_name = _get_hidden_layer_name(idx)
            with tf.name_scope(hidden_layer_name):
                layer = fully_connected(layer, n_units,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.05))

            self.layers[hidden_layer_name] = layer

        with tf.name_scope('output_layer'):
            self.logits = fully_connected(layer, 2)

        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.labels, self.logits))
