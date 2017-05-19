import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn_cell_impl import _RNNCell


class NestedLSTMWrapper(_RNNCell):
    def __init__(self, ref_cell, cells, temp, is_train, positions=None, opt=False, bias=None):
        if opt:
            assert not is_train
        assert isinstance(ref_cell, BasicLSTMCell)
        for cell in cells:
            assert isinstance(cell, BasicLSTMCell)

        self._ref_cell = ref_cell
        self._cells = cells
        self._is_train = is_train
        self._positions = [0] * len(cells) if positions is None else positions
        self._opt = opt
        self._num_cells = len(cells)
        self._state_size = ref_cell.state_size
        self._output_size = ref_cell.output_size + self._num_cells
        self._bias = tf.zeros([self._num_cells]) if bias is None else bias
        self._temp = temp

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope("nested_rnn_cell"):
            logits = tf.layers.dense(tf.concat([inputs] + list(state), 1), self._num_cells)  # [N, C]
            prob = tf.nn.softmax(logits)  # [N, C]
            logp = log(prob)
            choice = tf.argmax(prob + tf.expand_dims(self._bias, 0), axis=1)  # [N]

            def get_new_cell_state(i):
                with tf.variable_scope("cell_{}".format(i)) as vs:
                    cell = self._cells[i]
                    position = self._positions[i]
                    cell_state = LSTMStateTuple(*[tf.slice(each_state, [0, position], [-1, int(each_size)])
                                                  for each_state, each_size in zip(state, cell.state_size)])
                    _, new_cell_state = cell(inputs, cell_state)
                    new_cell_state = LSTMStateTuple(*[_merge(each_state, each_new_cell_state, position)
                                                      for each_state, each_new_cell_state in zip(state, new_cell_state)])
                    vs.reuse_variables()
                    return new_cell_state

            if self._opt:
                with tf.control_dependencies([tf.assert_equal(tf.shape(inputs)[0], 1)]):
                    def f0():
                        return tf.concat(get_new_cell_state(0), 1)
                    def f1():
                        return tf.concat(get_new_cell_state(1), 1)
                # return tf.concat(state, 1)
                    # Currently under development
                    assert self._num_cells == 2
                    choice = tf.reshape(choice, [])
                    # preds = [(tf.equal(choice, i), lambda: tf.concat(get_new_cell_state(i), 1))
                    #         for i in range(self._num_cells)]
                    preds = {tf.equal(choice, 0): f0, tf.equal(choice, 1): f1}
                    new_state = tf.case(preds, lambda: tf.concat(state, 1), exclusive=True)
                    new_state = tf.reshape(new_state, tf.shape(tf.concat(state, 1)))
                    # new_state = tf.cond(tf.equal(choice, 0), preds[0][1], preds[1][1])
                    new_state = LSTMStateTuple(*tf.split(new_state, 2, 1))
                    # new_state = state
            else:
                test_att = lambda: tf.one_hot(choice, self._num_cells, dtype='float')  # [N, C]
                train_att = lambda: tf.nn.softmax((logp + gumbel(tf.shape(logp))) / self._temp)  # [N, C]
                if isinstance(self._is_train, bool):
                    att = train_att() if self._is_train else test_att()
                else:
                    att = tf.cond(self._is_train, train_att, test_att)
                att = tf.expand_dims(att, -1)  # [N, C, 1]
                new_cell_states = [get_new_cell_state(i)
                                   for i in range(self._num_cells)]
                state_stack = _stack(new_cell_states, 1)  # [N, C, d] each
                new_state = LSTMStateTuple(*[tf.reduce_sum(each_state_stack * att, 1)
                                             for each_state_stack in state_stack])
            outputs = tf.concat([new_state[1], logp], 1)  # [N, d+C]
            return outputs, new_state

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size


def _merge(big_tensor, small_tensor, start):
    with tf.name_scope("merge"):
        a = tf.slice(big_tensor, [0, 0], [-1, start])
        b = tf.slice(big_tensor, [0, start + small_tensor.get_shape().as_list()[1]], [-1, -1])
        return tf.concat([a, small_tensor, b], 1)


def _stack(states, axis):
    with tf.name_scope("stack"):
        return states[0].__class__(*[tf.stack(each, axis=axis) for each in zip(*states)])


def gumbel(shape):
    return -log(-log(tf.random_uniform(shape)))


def log(x):
    return tf.log(x + 1e-12)
