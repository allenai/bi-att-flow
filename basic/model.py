import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from basic.read_data import DataSet
from my.tensorflow import exp_mask
from my.tensorflow.nn import linear


class Model(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N, M, JX, JQ, VW, VC = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size
        self.x = tf.placeholder('int32', [None, M, JX], name='x')
        self.x_mask = tf.placeholder('bool', [None, M, JX], name='x_mask')
        self.q = tf.placeholder('int32', [None, JQ], name='q')
        self.q_mask = tf.placeholder('bool', [None, JQ], name='q_mask')
        self.y = tf.placeholder('bool', [None, M, JX], name='y')
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
        N, M, JX, JQ, VW, VC, d = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size
        word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, d], dtype='float')

        Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d]
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        cell = BasicLSTMCell(d, state_is_tuple=True)

        Ax = tf.reshape(Ax, [-1, JX, d])
        x_len = tf.reshape(x_len, [-1])
        h, _ = dynamic_rnn(cell, Ax, x_len, dtype='float', scope='rnn/x')  # [N*M, JX, d]
        h = tf.reshape(h, [-1, M, JX, d])

        _, (_, u) = dynamic_rnn(cell, Aq, q_len, dtype='float', scope='rnn/q')  # [2, N, d]

        u = tf.expand_dims(tf.expand_dims(u, 1), 1)  # [N, 1, 1, d]
        self.logits = exp_mask(tf.reduce_sum(h * u, 3), self.x_mask)  # [N, M, JX]
        self.yp = tf.nn.sigmoid(self.logits)

    def _build_loss(self):
        ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, tf.cast(self.y, 'float')))
        tf.add_to_collection('losses', ce_loss)
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
        N, M, JX, JQ, VW, VC, d = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size
        feed_dict = {}

        x = np.zeros([N, M, JX], dtype='int32')
        x_mask = np.zeros([N, M, JX], dtype='bool')
        q = np.zeros([N, JQ], dtype='int32')
        q_mask = np.zeros([N, JQ], dtype='bool')

        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.q] = q
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.is_train] = is_train
        for i, xi in enumerate(batch.data['x']):
            for j, xij in enumerate(xi):
                for k, xijk in enumerate(xij):
                    x[i, j, k] = xijk
                    x_mask[i, j, k] = True

        for i, qi in enumerate(batch.data['q']):
            for j, qij in enumerate(qi):
                q[i, j] = qij
                q_mask[i, j] = True

        if supervised:
            y = np.zeros([N, M, JX], dtype='bool')
            feed_dict[self.y] = y
            for i, yi in enumerate(batch.data['y']):
                start_idx, stop_idx = yi
                j = start_idx[0]
                for k in range(start_idx[1], stop_idx[1]):
                    y[i, j, k] = True

        return feed_dict
