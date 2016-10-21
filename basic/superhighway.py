import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

from my.tensorflow.nn import linear


class SHCell(RNNCell):
    """
    Super-Highway Cell
    """
    def __init__(self, input_size):
        self._state_size = input_size
        self._output_size = input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "SHCell"):
            h, u = tf.split(1, 2, inputs)
            a = tf.nn.sigmoid(linear([h * u], self._state_size, True))
            new_state = a * state + (1 - a) * h
            outputs = new_state
            return outputs, new_state

