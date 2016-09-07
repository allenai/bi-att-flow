import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from prob.read_data import DataSet
from my.tensorflow import exp_mask, get_initializer
from my.tensorflow import mask
from my.tensorflow.nn import linear, double_linear_logits, linear_logits, softsel
from my.tensorflow.rnn import bidirectional_dynamic_rnn, dynamic_rnn
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
        self.y = tf.placeholder('bool', [None, M, JX], name='y')
        self.y2 = tf.placeholder('bool', [None, M, JX], name='y2')
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
            Acx = tf.nn.dropout(Acx, config.input_keep_prob)
            Acq = tf.nn.dropout(Acq, config.input_keep_prob)

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
            Ax = linear([Ax], d, False, scope='Ax_reshape', wd=config.wd, input_keep_prob=config.input_keep_prob,
                        is_train=self.is_train)
            Aq = linear([Aq], d, False, scope='Aq_reshape', wd=config.wd, input_keep_prob=config.input_keep_prob,
                        is_train=self.is_train)

        xx = tf.concat(3, [xxc, Ax])  # [N, M, JX, 2d]
        qq = tf.concat(2, [qqc, Aq])  # [N, JQ, 2d]

        cell = BasicLSTMCell(d, state_is_tuple=True)
        cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("prepro"):
            _, (_, (fw_u, bw_u)) = bidirectional_dynamic_rnn(cell, cell, qq, q_len, dtype='float', scope='u')  # [N, J, d], [N, d]
            u = tf.concat(1, [fw_u, bw_u])
            (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='h')  # [N, M, JX, 2d]
            h = tf.concat(3, [fw_h, bw_h])

        with tf.variable_scope("main"):
            u = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1])
            p0 = tf.concat(3, [h, u, h*u, tf.abs(h-u)])
            (fw_g1, bw_g1), _ = bidirectional_dynamic_rnn(cell, cell, p0, x_len, dtype='float', scope='h1')  # [N, M, JX, 2d]
            g1 = tf.concat(3, [fw_g1, bw_g1])
            dot = double_linear_logits(g1, d, True, mask=self.x_mask, wd=config.wd, input_keep_prob=config.input_keep_prob, is_train=self.is_train, scope='logits1')
            hyp1 = tf.reshape(tf.one_hot(tf.argmax(tf.reshape(dot, [N, M*JX]), 1), M*JX), [N, M, JX])
            train_g1i = tf.reduce_sum(mask(g1, tf.expand_dims(self.y, -1)), [1, 2])
            test_g1i = tf.reduce_sum(mask(g1, tf.expand_dims(hyp1, -1)), [1, 2])
            # g1i = softsel(tf.reshape(g1, [N, M*JX, 2*d]), tf.reshape(dot, [N, M*JX]))
            g1i = tf.cond(self.is_train, lambda: train_g1i, lambda: test_g1i)
            g1i_tiled = tf.tile(tf.expand_dims(tf.expand_dims(g1i, 1), 1), [1, M, JX, 1])
            p1 = tf.concat(3, [g1, g1i_tiled, g1 * g1i_tiled, tf.abs(g1 - g1i_tiled)])
            p1 = g1
            # g1 = tf.concat(3, [g1, u, g1*u, tf.abs(g1-u)])
            (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(cell, cell, p1, x_len, dtype='float', scope='h2')  # [N, M, JX, 2d]
            g2 = tf.concat(3, [fw_g2, bw_g2])
            dot2 = double_linear_logits(g2, d, True, mask=self.x_mask, wd=config.wd, input_keep_prob=config.input_keep_prob, is_train=self.is_train, scope='logits2')
            hyp2 = tf.reshape(tf.one_hot(tf.argmax(tf.reshape(dot2, [N, M*JX]), 1), M*JX), [N, M, JX])
            # g2 = tf.concat(3, [g2, u, g2*u, tf.abs(g2-u)])

        """
        dot = linear_logits(g1, True, scope='dot', wd=config.wd, input_keep_prob=config.input_keep_prob,
                            is_train=self.is_train)
        dot2 = linear_logits(g2, True, scope='dot2', wd=config.wd, input_keep_prob=config.input_keep_prob,
                             is_train=self.is_train)
        """
        self.logits = tf.reshape(dot, [-1, M * JX])  # [N, M, JX]
        self.logits2 = tf.reshape(dot2, [-1, M * JX])

        self.hyp1 = hyp1
        self.hyp2 = hyp2
        self.yp = tf.reshape(tf.nn.softmax(self.logits), [-1, M, JX])
        self.yp2 = tf.reshape(tf.nn.softmax(self.logits2), [-1, M, JX])

    def _build_loss(self):
        config = self.config
        N, M, JX, JQ, VW, VC = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.logits, tf.cast(tf.reshape(self.y, [-1, M * JX]), 'float')))
        tf.add_to_collection('losses', ce_loss)
        ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.logits2, tf.cast(tf.reshape(self.y2, [-1, M * JX]), 'float')))
        tf.add_to_collection("losses", ce_loss2)

        self.loss = tf.add_n(tf.get_collection('losses'), name='loss')
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
                if j == config.max_num_sents:
                    break
                for k, xijk in enumerate(xij):
                    if k == config.max_sent_size:
                        break
                    x[i, j, k] = _get_word(xijk)
                    x_mask[i, j, k] = True

        for i, cxi in enumerate(batch.data['cx']):
            for j, cxij in enumerate(cxi):
                if j == config.max_num_sents:
                    break
                for k, cxijk in enumerate(cxij):
                    if k == config.max_sent_size:
                        break
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        cx[i, j, k, l] = _get_char(cxijkl)

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
            y = np.zeros([N, M, JX], dtype='bool')
            y2 = np.zeros([N, M, JX], dtype='bool')
            feed_dict[self.y] = y
            feed_dict[self.y2] = y2
            for i, yi in enumerate(batch.data['y']):
                start_idx, stop_idx = random.choice(yi)
                j, k = start_idx
                y[i, j, k] = True
                j2, k2 = stop_idx
                y2[i, j2, k2-1] = True

        return feed_dict
