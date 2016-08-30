import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from memnn.read_data import DataSet
from my.tensorflow import exp_mask, get_initializer
from my.tensorflow.nn import linear
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper


class Model(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N, M, JX, JQ, VW, VC, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
        self.x = tf.placeholder('int32', [None, M, JX], name='x')
        self.cx = tf.placeholder('int32', [None, M, JX, W], name='cx')
        self.x_mask = tf.placeholder('bool', [None, M, JX], name='x_mask')
        self.q = tf.placeholder('int32', [None, JQ], name='q')
        self.cq = tf.placeholder('int32', [None, JQ, W], name='cq')
        self.q_mask = tf.placeholder('bool', [None, JQ], name='q_mask')
        self.sy = tf.placeholder('int32', [None], name='sy')
        self.wy = tf.placeholder('int32', [None], name='wy')
        self.is_train = tf.placeholder('bool', [], name='is_train')

        # Define misc

        # Forward outputs / loss inputs
        self.s_logits = None
        self.w_logits = None
        self.syp = None
        self.wyp = None
        self.var_list = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()

        self.ema_op = self._get_ema_op()
        self.summary = tf.merge_all_summaries()

    def _build_forward(self):
        config = self.config
        N, M, JX, JQ, VW, VC, d, dc, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.char_emb_size, config.max_word_size

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

        xx = tf.concat(3, [xxc, Ax])  # [N, M, JX, 2d]
        qq = tf.concat(2, [qqc, Aq])  # [N, JQ, 2d]
        D = d + config.word_emb_size

        cell = BasicLSTMCell(d, state_is_tuple=True)
        cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("rnn1"):
            (fw_hs, bw_hs), (_, (fw_h, bw_h)) = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='start')  # [N, M, JX, 2d], [N, M, 2d]
            tf.get_variable_scope().reuse_variables()
            (fw_us, bw_us), (_, (fw_u, bw_u)) = bidirectional_dynamic_rnn(cell, cell, qq, q_len, dtype='float', scope='start')  # [N, J, d], [N, d]
            hs = tf.concat(3, [fw_hs, bw_hs])
            h = tf.reshape(tf.concat(1, (fw_h, bw_h)), [N, M, 2*d])
            u = tf.concat(1, [fw_u, bw_u])

        u = tf.expand_dims(u, 1)  # [N 1, 4d]
        s_dot = linear(h * u, 1, True, squeeze=True, scope='s_dot', wd=config.wd)
        hsi = tf.reduce_sum(hs * tf.expand_dims(tf.expand_dims(tf.one_hot(self.sy, M), -1), -1), 1)
        w_dot = linear(hsi * u, 1, True, squeeze=True, scope='w_dot', wd=config.wd)
        self.s_logits = exp_mask(s_dot, tf.reduce_any(self.x_mask, 2))  # [N, M,]
        x_mask_i = tf.reduce_any(self.x_mask & tf.expand_dims(tf.one_hot(self.sy, M, on_value=True, off_value=False), -1), 1)
        self.w_logits = exp_mask(w_dot, x_mask_i)
        self.syp = tf.nn.softmax(self.s_logits)
        self.wyp = tf.nn.softmax(self.w_logits)

    def _build_loss(self):
        config = self.config
        N, M, JX, JQ, VW, VC = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size
        # s_ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.s_logits, self.sy), name='s_ce_loss')
        w_ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.w_logits, self.wy), name='w_ce_loss')
        # tf.add_to_collection('losses', s_ce_loss)
        tf.add_to_collection('losses', w_ce_loss)

        self.loss = tf.add_n(tf.get_collection('losses'), name='loss')
        # tf.scalar_summary(s_ce_loss.op.name, s_ce_loss)
        tf.scalar_summary(w_ce_loss.op.name, w_ce_loss)
        tf.scalar_summary(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    def _get_ema_op(self):
        ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema_op = ema.apply(tf.get_collection("ema/scalar") + tf.get_collection("ema/histogram"))
        for var in tf.get_collection("ema/scalar"):
            ema_var = ema.average(var)
            tf.scalar_summary(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/histogram"):
            ema_var = ema.average(var)
            tf.histogram_summary(ema_var.op.name, ema_var)
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
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        x = np.zeros([N, M, JX], dtype='int32')
        cx = np.zeros([N, M, JX, W], dtype='int32')
        x_mask = np.zeros([N, M, JX], dtype='bool')
        q = np.zeros([N, JQ], dtype='int32')
        cq = np.zeros([N, JQ, W], dtype='int32')
        q_mask = np.zeros([N, JQ], dtype='bool')

        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        feed_dict[self.cq] = cq
        feed_dict[self.q_mask] = q_mask
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

        for i, xi in enumerate(batch.data['x']):
            for j, xij in enumerate(xi):
                for k, xijk in enumerate(xij):
                    x[i, j, k] = _get_word(xijk)
                    x_mask[i, j, k] = True

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
                q_mask[i, j] = True

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        if supervised:
            sy = np.zeros([N], dtype='int32')
            wy = np.zeros([N], dtype='int32')
            feed_dict[self.sy] = sy
            feed_dict[self.wy] = wy
            for i, yi in enumerate(batch.data['y']):
                start_idx, stop_idx = yi
                sy[i], wy[i] = start_idx

        return feed_dict
