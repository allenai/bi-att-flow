import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper, RNNCell

from my.tensorflow import exp_mask


class SwitchableDropoutWrapper(DropoutWrapper):
    def __init__(self, cell, is_train, input_keep_prob=1.0, output_keep_prob=1.0,
             seed=None):
        super(SwitchableDropoutWrapper, self).__init__(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob,
                                                       seed=seed)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        outputs_do, _ = super(SwitchableDropoutWrapper, self).__call__(inputs, state, scope=scope)
        tf.get_variable_scope().reuse_variables()
        outputs, new_state = self._cell(inputs, state, scope)
        outputs = tf.cond(self.is_train, lambda: outputs_do, lambda: outputs)
        return outputs, new_state


class TreeRNNCell(RNNCell):
    def __init__(self, cell, input_size, reduce_func):
        self._cell = cell
        self._input_size = input_size
        self._reduce_func = reduce_func

    def __call__(self, inputs, state, scope=None):
        """
        :param inputs: [N*B, I + B]
        :param state: [N*B, d]
        :param scope:
        :return: [N*B, d]
        """
        with tf.variable_scope(scope or self.__class__.__name__):
            d = self.state_size
            x = tf.slice(inputs, [0, 0], [-1, self._input_size])  # [N*B, I]
            mask = tf.slice(inputs, [0, self._input_size], [-1, -1])  # [N*B, B]
            B = tf.shape(mask)[1]
            prev_state = tf.expand_dims(tf.reshape(state, [-1, B, d]), 1)  # [N, B, d] -> [N, 1, B, d]
            mask = tf.tile(tf.expand_dims(tf.reshape(mask, [-1, B, B]), -1), [1, 1, 1, d])  # [N, B, B, d]
            # prev_state = self._reduce_func(tf.tile(prev_state, [1, B, 1, 1]), 2)
            prev_state = self._reduce_func(exp_mask(prev_state, mask), 2)  # [N, B, d]
            prev_state = tf.reshape(prev_state, [-1, d])  # [N*B, d]
            return self._cell(x, prev_state)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size


class NoOpCell(RNNCell):
    def __init__(self, num_units):
        self._num_units = num_units

    def __call__(self, inputs, state, scope=None):
        return state, state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units
