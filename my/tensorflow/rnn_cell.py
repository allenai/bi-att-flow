import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, RNNCell, LSTMStateTuple

from my.tensorflow import exp_mask, flatten
from my.tensorflow.nn import linear, softsel, double_linear_logits


class SwitchableDropoutWrapper(DropoutWrapper):
    def __init__(self, cell, is_train, input_keep_prob=1.0, output_keep_prob=1.0,
             seed=None):
        super(SwitchableDropoutWrapper, self).__init__(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob,
                                                       seed=seed)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        outputs_do, new_state_do = super(SwitchableDropoutWrapper, self).__call__(inputs, state, scope=scope)
        tf.get_variable_scope().reuse_variables()
        outputs, new_state = self._cell(inputs, state, scope)
        outputs = tf.cond(self.is_train, lambda: outputs_do, lambda: outputs)
        if isinstance(state, tuple):
            new_state = state.__class__(*[tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                                          for new_state_do_i, new_state_i in zip(new_state_do, new_state)])
        else:
            new_state = tf.cond(self.is_train, lambda: new_state_do, lambda: new_state)
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


class MatchCell(RNNCell):
    def __init__(self, cell, input_size, q_len):
        self._cell = cell
        self._input_size = input_size
        # FIXME : This won't be needed with good shape guessing
        self._q_len = q_len

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """

        :param inputs: [N, d + JQ + JQ * d]
        :param state: [N, d]
        :param scope:
        :return:
        """
        with tf.variable_scope(scope or self.__class__.__name__):
            c_prev, h_prev = state
            x = tf.slice(inputs, [0, 0], [-1, self._input_size])
            q_mask = tf.slice(inputs, [0, self._input_size], [-1, self._q_len])  # [N, JQ]
            qs = tf.slice(inputs, [0, self._input_size + self._q_len], [-1, -1])
            qs = tf.reshape(qs, [-1, self._q_len, self._input_size])  # [N, JQ, d]
            x_tiled = tf.tile(tf.expand_dims(x, 1), [1, self._q_len, 1])  # [N, JQ, d]
            h_prev_tiled = tf.tile(tf.expand_dims(h_prev, 1), [1, self._q_len, 1])  # [N, JQ, d]
            f = tf.tanh(linear([qs, x_tiled, h_prev_tiled], self._input_size, True, scope='f'))  # [N, JQ, d]
            a = tf.nn.softmax(exp_mask(linear(f, 1, True, squeeze=True, scope='a'), q_mask))  # [N, JQ]
            q = tf.reduce_sum(qs * tf.expand_dims(a, -1), 1)
            z = tf.concat(axis=1, values=[x, q])  # [N, 2d]
            return self._cell(z, state)


class AttentionCell(RNNCell):
    def __init__(self, cell, memory, mask=None, controller=None, mapper=None, input_keep_prob=1.0, is_train=None):
        """
        Early fusion attention cell: uses the (inputs, state) to control the current attention.

        :param cell:
        :param memory: [N, M, m]
        :param mask:
        :param controller: (inputs, prev_state, memory) -> memory_logits
        """
        self._cell = cell
        self._memory = memory
        self._mask = mask
        self._flat_memory = flatten(memory, 2)
        self._flat_mask = flatten(mask, 1)
        if controller is None:
            controller = AttentionCell.get_linear_controller(True, is_train=is_train)
        self._controller = controller
        if mapper is None:
            mapper = AttentionCell.get_concat_mapper()
        elif mapper == 'sim':
            mapper = AttentionCell.get_sim_mapper()
        self._mapper = mapper

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "AttentionCell"):
            memory_logits = self._controller(inputs, state, self._flat_memory)
            sel_mem = softsel(self._flat_memory, memory_logits, mask=self._flat_mask)  # [N, m]
            new_inputs, new_state = self._mapper(inputs, state, sel_mem)
            return self._cell(new_inputs, state)

    @staticmethod
    def get_double_linear_controller(size, bias, input_keep_prob=1.0, is_train=None):
        def double_linear_controller(inputs, state, memory):
            """

            :param inputs: [N, i]
            :param state: [N, d]
            :param memory: [N, M, m]
            :return: [N, M]
            """
            rank = len(memory.get_shape())
            _memory_size = tf.shape(memory)[rank-2]
            tiled_inputs = tf.tile(tf.expand_dims(inputs, 1), [1, _memory_size, 1])
            if isinstance(state, tuple):
                tiled_states = [tf.tile(tf.expand_dims(each, 1), [1, _memory_size, 1])
                                for each in state]
            else:
                tiled_states = [tf.tile(tf.expand_dims(state, 1), [1, _memory_size, 1])]

            # [N, M, d]
            in_ = tf.concat([tiled_inputs] + tiled_states + [memory], axis=2)
            out = double_linear_logits(in_, size, bias, input_keep_prob=input_keep_prob,
                                       is_train=is_train)
            return out
        return double_linear_controller

    @staticmethod
    def get_linear_controller(bias, input_keep_prob=1.0, is_train=None):
        def linear_controller(inputs, state, memory):
            rank = len(memory.get_shape())
            _memory_size = tf.shape(memory)[rank-2]
            tiled_inputs = tf.tile(tf.expand_dims(inputs, 1), [1, _memory_size, 1])
            if isinstance(state, tuple):
                tiled_states = [tf.tile(tf.expand_dims(each, 1), [1, _memory_size, 1])
                                for each in state]
            else:
                tiled_states = [tf.tile(tf.expand_dims(state, 1), [1, _memory_size, 1])]

            # [N, M, d]
            in_ = tf.concat([tiled_inputs] + tiled_states + [memory], axis=2)
            out = linear(in_, 1, bias, squeeze=True, input_keep_prob=input_keep_prob, is_train=is_train)
            return out
        return linear_controller

    @staticmethod
    def get_concat_mapper():
        def concat_mapper(inputs, state, sel_mem):
            """

            :param inputs: [N, i]
            :param state: [N, d]
            :param sel_mem: [N, m]
            :return: (new_inputs, new_state) tuple
            """
            return tf.concat(axis=1, values=[inputs, sel_mem]), state
        return concat_mapper

    @staticmethod
    def get_sim_mapper():
        def sim_mapper(inputs, state, sel_mem):
            """
            Assume that inputs and sel_mem are the same size
            :param inputs: [N, i]
            :param state: [N, d]
            :param sel_mem: [N, i]
            :return: (new_inputs, new_state) tuple
            """
            return tf.concat(axis=1, values=[inputs, sel_mem, inputs * sel_mem, tf.abs(inputs - sel_mem)]), state
        return sim_mapper
