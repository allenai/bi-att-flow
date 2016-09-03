import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from match.read_data import DataSet
from my.tensorflow import exp_mask, get_initializer
from my.tensorflow.nn import linear
from my.tensorflow.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, MatchCell


class Model(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N, JX, JQ, VW, VC, W = \
            config.batch_size, config.max_para_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
        self.x = tf.placeholder('int32', [None, JX], name='x')
        self.cx = tf.placeholder('int32', [None, JX, W], name='cx')
        self.x_mask = tf.placeholder('bool', [None, JX], name='x_mask')
        self.q = tf.placeholder('int32', [None, JQ], name='q')
        self.cq = tf.placeholder('int32', [None, JQ, W], name='cq')
        self.q_mask = tf.placeholder('bool', [None, JQ], name='q_mask')
        self.y = tf.placeholder('bool', [None, JX], name='y')
        self.y2 = tf.placeholder('bool', [None, JX], name='y2')
        self.is_train = tf.placeholder('bool', [], name='is_train')

        # Define misc

        # Forward outputs / loss inputs
        self.logits = None
        self.yp = None
        self.yp2 = None
        self.var_list = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()

        self.ema_op = self._get_ema_op()
        self.summary = tf.merge_all_summaries()

    def _build_forward(self):
        config = self.config
        N, JX, JQ, VW, VC, d, dc, W = \
            config.batch_size, config.max_para_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.char_emb_size, config.max_word_size

        with tf.variable_scope("char_emb"):
            char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')
            Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, JX, W, dc]
            Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]

            filter = tf.get_variable("filter", shape=[1, config.char_filter_height, dc, d], dtype='float')
            bias = tf.get_variable("bias", shape=[d], dtype='float')
            strides = [1, 1, 1, 1]
            Acx = tf.reshape(Acx, [-1, JX, W, dc])
            Acq = tf.reshape(Acq, [-1, JQ, W, dc])
            xxc = tf.nn.conv2d(Acx, filter, strides, "VALID") + bias  # [N, JX, W/filter_stride, d]
            qqc = tf.nn.conv2d(Acq, filter, strides, "VALID") + bias  # [N, JQ, W/filter_stride, d]
            xxc = tf.reshape(tf.reduce_max(tf.nn.relu(xxc), 2), [-1, JX, d])
            qqc = tf.reshape(tf.reduce_max(tf.nn.relu(qqc), 2), [-1, JQ, d])

        with tf.variable_scope("word_emb"):
            if config.mode == 'train':
                word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, config.word_emb_size], initializer=get_initializer(config.emb_mat))
            else:
                word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, config.word_emb_size], dtype='float')
            Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, JX, d]
            Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]
            # Ax = linear([Ax], d, False, scope='Ax_reshape')
            # Aq = linear([Aq], d, False, scope='Aq_reshape')

        xx = tf.concat(2, [xxc, Ax])  # [N, JX, 2d]
        qq = tf.concat(2, [qqc, Aq])  # [N, JQ, 2d]
        # xx = Ax
        # qq = Aq

        cell = BasicLSTMCell(d, state_is_tuple=True)
        cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 1)  # [N]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("prepro"):
            _, (_, ul) = dynamic_rnn(cell, qq, q_len, dtype='float', scope='common')  # [N, J, d], [N, d]
            tf.get_variable_scope().reuse_variables()
            h, _ = dynamic_rnn(cell, xx, x_len, dtype='float', scope='common')  # [N, JX, 2d]

        with tf.variable_scope("inference"):
            ul = tf.tile(tf.expand_dims(ul, 1), [1, JX, 1])
            hul = tf.concat(2, [h, ul])
            (fw_g1, bw_g1), _ = bidirectional_dynamic_rnn(cell, cell, hul, x_len, dtype='float', scope='h12')  # [N, JX, 2d]
            g1ul = tf.concat(2, [fw_g1, bw_g1, ul])
            (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(cell, cell, g1ul, x_len, dtype='float', scope='h2')  # [N, JX, 2d]
            g2 = tf.concat(2, [fw_g2, bw_g2])

        with tf.variable_scope("mlp"):
            dot1 = linear(g1ul, 1, True, squeeze=True, scope='dot1', wd=config.wd)
            dot2 = linear(g2, 1, True, squeeze=True, scope='dot2', wd=config.wd)
        self.logits = exp_mask(dot1, self.x_mask)
        self.logits2 = exp_mask(dot2, self.x_mask)
        self.yp = tf.nn.softmax(self.logits)
        self.yp2 = tf.nn.softmax(self.logits2)

        """
        with tf.variable_scope("start_match"):
            match_cell = MatchCell(cell, 2*d, JQ)
            q_mask_tiled = tf.tile(tf.reshape(self.q_mask, [N, 1, JQ]), [1, JX, 1])
            u_tiled = tf.tile(tf.reshape(u, [N, 1, JQ*2*d]), [1, JX, 1])
            hu = tf.concat(2, [h, tf.cast(q_mask_tiled, 'float'), u_tiled])  # [N, JX, 2d + JQ + JQ*d]
            (fw_hr, bw_hr), _ = bidirectional_dynamic_rnn(match_cell, match_cell, hu, x_len, dtype='float', scope='hr')
            hr = tf.concat(2, [fw_hr, bw_hr])  # [N, JX, 2*d]
            dot = linear(hr, 1, True, squeeze=True, scope='dot', wd=config.wd)

        with tf.variable_scope("stop_match"):
            (fw_hr2, bw_hr2), _ = bidirectional_dynamic_rnn(cell, cell, hr, x_len, dtype='float', scope='hr2')
            hr2 = tf.concat(2, [fw_hr2, bw_hr2])
            dot2 = linear(hr2, 1, True, squeeze=True, scope='dot2', wd=config.wd)

        self.logits = exp_mask(dot, self.x_mask)  # [N, M, JX]
        self.logits2 = exp_mask(dot2, self.x_mask)
        self.yp = tf.nn.softmax(self.logits)
        self.yp2 = tf.nn.softmax(self.logits2)
        """

    def _build_loss(self):
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.logits, tf.cast(self.y, 'float')))
        tf.add_to_collection('losses', ce_loss)
        ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.logits2, tf.cast(self.y2, 'float')))
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
        N, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_para_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        x = np.zeros([N, JX], dtype='int32')
        cx = np.zeros([N, JX, W], dtype='int32')
        x_mask = np.zeros([N, JX], dtype='bool')
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
            cur_idx = 0
            for j, xij in enumerate(xi):
                for k, xijk in enumerate(xij):
                    x[i, cur_idx] = _get_word(xijk)
                    x_mask[i, cur_idx] = True
                    cur_idx += 1

        for i, cxi in enumerate(batch.data['cx']):
            cur_idx = 0
            for j, cxij in enumerate(cxi):
                for k, cxijk in enumerate(cxij):
                    for l, cxijkl in enumerate(cxijk):
                        cx[i, cur_idx, l] = _get_char(cxijkl)
                        if l + 1 == config.max_word_size:
                            break
                    cur_idx += 1

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
            y = np.zeros([N, JX], dtype='bool')
            y2 = np.zeros([N, JX], dtype='bool')
            feed_dict[self.y] = y
            feed_dict[self.y2] = y2
            for i, (xi, yi) in enumerate(zip(batch.data['x'], batch.data['y'])):
                start_idx, stop_idx = random.choice(yi)
                j, k = start_idx
                idx = sum(len(xi[jj]) for jj in range(j)) + k
                y[i, idx] = True
                j2, k2 = stop_idx
                idx = sum(len(xi[jj2]) for jj2 in range(j2)) + k2 - 1
                y2[i, idx] = True

        return feed_dict
