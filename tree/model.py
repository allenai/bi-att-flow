import nltk
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from my.nltk_utils import tree2matrix, find_max_f1_subtree, load_compressed_tree, set_span
from tree.read_data import DataSet
from my.tensorflow import exp_mask, get_initializer
from my.tensorflow.nn import linear
from my.tensorflow.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, NoOpCell, TreeRNNCell


class Model(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N, M, JX, JQ, VW, VC, W, H = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size, config.max_tree_height
        self.x = tf.placeholder('int32', [None, M, JX], name='x')
        self.cx = tf.placeholder('int32', [None, M, JX, W], name='cx')
        self.q = tf.placeholder('int32', [None, JQ], name='q')
        self.cq = tf.placeholder('int32', [None, JQ, W], name='cq')
        self.tx = tf.placeholder('int32', [None, M, H, JX], name='tx')
        self.tx_edge_mask = tf.placeholder('bool', [None, M, H, JX, JX], name='tx_edge_mask')
        self.y = tf.placeholder('bool', [None, M, H, JX], name='y')
        self.is_train = tf.placeholder('bool', [], name='is_train')

        # Define misc

        # Forward outputs / loss inputs
        self.logits = None
        self.yp = None
        self.var_list = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()

        self.ema_op = self._get_ema_op()
        self.summary = tf.summary.merge_all()

    def _build_forward(self):
        config = self.config
        N, M, JX, JQ, VW, VC, d, dc, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.char_emb_size, config.max_word_size
        H = config.max_tree_height

        x_mask = self.x > 0
        q_mask = self.q > 0
        tx_mask = self.tx > 0  # [N, M, H, JX]

        with tf.variable_scope("char_emb"):
            char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')
            Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, M, JX, W, dc]
            Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]

            filter = tf.get_variable("filter", shape=[1, config.char_filter_height, dc, d], dtype='float')
            bias = tf.get_variable("bias", shape=[d], dtype='float')
            strides = [1, 1, 1, 1]
            Acx = tf.reshape(Acx, [-1, JX, W, dc])
            Acq = tf.reshape(Acq, [-1, JQ, W, dc])
            xxc = tf.nn.conv2d(Acx, filter, strides, "VALID") + bias  # [N*M, JX, W/filter_stride, d]
            qqc = tf.nn.conv2d(Acq, filter, strides, "VALID") + bias  # [N, JQ, W/filter_stride, d]
            xxc = tf.reshape(tf.reduce_max(tf.nn.relu(xxc), 2), [-1, M, JX, d])
            qqc = tf.reshape(tf.reduce_max(tf.nn.relu(qqc), 2), [-1, JQ, d])

        with tf.variable_scope("word_emb"):
            if config.mode == 'train':
                word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, config.word_emb_size], initializer=get_initializer(config.emb_mat))
            else:
                word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, config.word_emb_size], dtype='float')
            Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d]
            Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]
            # Ax = linear([Ax], d, False, scope='Ax_reshape')
            # Aq = linear([Aq], d, False, scope='Aq_reshape')

        xx = tf.concat(axis=3, values=[xxc, Ax])  # [N, M, JX, 2d]
        qq = tf.concat(axis=2, values=[qqc, Aq])  # [N, JQ, 2d]
        D = d + config.word_emb_size

        with tf.variable_scope("pos_emb"):
            pos_emb_mat = tf.get_variable("pos_emb_mat", shape=[config.pos_vocab_size, d], dtype='float')
            Atx = tf.nn.embedding_lookup(pos_emb_mat, self.tx)  # [N, M, H, JX, d]

        cell = BasicLSTMCell(D, state_is_tuple=True)
        cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        x_len = tf.reduce_sum(tf.cast(x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("rnn"):
            (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='start')  # [N, M, JX, 2d]
            tf.get_variable_scope().reuse_variables()
            (fw_us, bw_us), (_, (fw_u, bw_u)) = bidirectional_dynamic_rnn(cell, cell, qq, q_len, dtype='float', scope='start')  # [N, J, d], [N, d]
            u = (fw_u + bw_u) / 2.0
            h = (fw_h + bw_h) / 2.0

        with tf.variable_scope("h"):
            no_op_cell = NoOpCell(D)
            tree_rnn_cell = TreeRNNCell(no_op_cell, d, tf.reduce_max)
            initial_state = tf.reshape(h, [N*M*JX, D])  # [N*M*JX, D]
            inputs = tf.concat(axis=4, values=[Atx, tf.cast(self.tx_edge_mask, 'float')])  # [N, M, H, JX, d+JX]
            inputs = tf.reshape(tf.transpose(inputs, [0, 1, 3, 2, 4]), [N*M*JX, H, d + JX])  # [N*M*JX, H, d+JX]
            length = tf.reshape(tf.reduce_sum(tf.cast(tx_mask, 'int32'), 2), [N*M*JX])
            # length = tf.reshape(tf.reduce_sum(tf.cast(tf.transpose(tx_mask, [0, 1, 3, 2]), 'float'), 3), [-1])
            h, _ = dynamic_rnn(tree_rnn_cell, inputs, length, initial_state=initial_state)  # [N*M*JX, H, D]
            h = tf.transpose(tf.reshape(h, [N, M, JX, H, D]), [0, 1, 3, 2, 4])  # [N, M, H, JX, D]

        u = tf.expand_dims(tf.expand_dims(tf.expand_dims(u, 1), 1), 1)  # [N, 1, 1, 1, 4d]
        dot = linear(h * u, 1, True, squeeze=True, scope='dot')  # [N, M, H, JX]
        # self.logits = tf.reshape(dot, [N, M * H * JX])
        self.logits = tf.reshape(exp_mask(dot, tx_mask), [N, M * H * JX])  # [N, M, H, JX]
        self.yp = tf.reshape(tf.nn.softmax(self.logits), [N, M, H, JX])

    def _build_loss(self):
        config = self.config
        N, M, JX, JQ, VW, VC = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size
        H = config.max_tree_height
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=tf.cast(tf.reshape(self.y, [N, M * H * JX]), 'float')))
        tf.add_to_collection('losses', ce_loss)
        self.loss = tf.add_n(tf.get_collection('losses'), name='loss')
        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    def _get_ema_op(self):
        ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema_op = ema.apply(tf.get_collection("ema/scalar") + tf.get_collection("ema/histogram"))
        for var in tf.get_collection("ema/scalar"):
            ema_var = ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/histogram"):
            ema_var = ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)
        return ema_op

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        assert isinstance(batch, DataSet)
        config = self.config
        N, M, JX, JQ, VW, VC, d, W, H = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size, \
            config.max_tree_height
        feed_dict = {}

        x = np.zeros([N, M, JX], dtype='int32')
        cx = np.zeros([N, M, JX, W], dtype='int32')
        q = np.zeros([N, JQ], dtype='int32')
        cq = np.zeros([N, JQ, W], dtype='int32')
        tx = np.zeros([N, M, H, JX], dtype='int32')
        tx_edge_mask = np.zeros([N, M, H, JX, JX], dtype='bool')

        feed_dict[self.x] = x
        feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        feed_dict[self.cq] = cq
        feed_dict[self.tx] = tx
        feed_dict[self.tx_edge_mask] = tx_edge_mask
        feed_dict[self.is_train] = is_train

        def _get_word(word):
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            return 1

        def _get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1

        def _get_pos(tree):
            d = batch.shared['pos2idx']
            if tree.label() in d:
                return d[tree.label()]
            return 1

        for i, xi in enumerate(batch.data['x']):
            for j, xij in enumerate(xi):
                for k, xijk in enumerate(xij):
                    x[i, j, k] = _get_word(xijk)

        for i, cxi in enumerate(batch.data['cx']):
            for j, cxij in enumerate(cxi):
                for k, cxijk in enumerate(cxij):
                    for l, cxijkl in enumerate(cxijk):
                        cx[i, j, k, l] = _get_char(cxijkl)
                        if l + 1 == config.max_word_size:
                            break

        for i, qi in enumerate(batch.data['q']):
            for j, qij in enumerate(qi):
                q[i, j] = _get_word(qij)

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        for i, txi in enumerate(batch.data['stx']):
            for j, txij in enumerate(txi):
                txij_mat, txij_mask = tree2matrix(nltk.tree.Tree.fromstring(txij), _get_pos, row_size=H, col_size=JX)
                tx[i, j, :, :], tx_edge_mask[i, j, :, :, :] = txij_mat, txij_mask

        if supervised:
            y = np.zeros([N, M, H, JX], dtype='bool')
            feed_dict[self.y] = y
            for i, yi in enumerate(batch.data['y']):
                start_idx, stop_idx = yi
                sent_idx = start_idx[0]
                if start_idx[0] == stop_idx[0]:
                    span = [start_idx[1], stop_idx[1]]
                else:
                    span = [start_idx[1], len(batch.data['x'][sent_idx])]
                tree = nltk.tree.Tree.fromstring(batch.data['stx'][i][sent_idx])
                set_span(tree)
                best_subtree = find_max_f1_subtree(tree, span)

                def _get_y(t):
                    return t == best_subtree

                yij, _ = tree2matrix(tree, _get_y, H, JX, dtype='bool')
                y[i, sent_idx, :, :] = yij

        return feed_dict
