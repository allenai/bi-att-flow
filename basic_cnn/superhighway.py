import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

from my.tensorflow.nn import linear


class SHCell(RNNCell):
    """
    Super-Highway Cell
    """
    def __init__(self, input_size, logit_func='tri_linear'):
        self._state_size = input_size
        self._output_size = input_size
        self._logit_func = logit_func

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "SHCell"):
            h, u = tf.split(1, 2, inputs)
            if self._logit_func == 'mul_linear':
                args = [h * u]
            elif self._logit_func == 'linear':
                args = [h, u]
            elif self._logit_func == 'tri_linear':
                args = [h, u, h * u]
            else:
                raise Exception()
            a = tf.nn.sigmoid(linear(args, self._state_size, True))
            new_state = a * state + (1 - a) * h
            outputs = new_state
            return outputs, new_state

