import os

import numpy as np
import tensorflow as tf

from basic.read_data import DataSet
from my.tensorflow.nn import linear


class Model(object):
    def __init__(self, config):
        self.config = config
        self.writer = None
        self.saver = None
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)
        # Define forward inputs here
        self.x = tf.placeholder('float', [config.batch_size, config.dim], name='x')
        self.y = tf.placeholder('int32', [config.batch_size], name='y')

        # Forward outputs / loss inputs
        self.logits = None
        self.yp = None
        self.var_list = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()

        self.summary = tf.merge_all_summaries()

    def _build_forward(self):
        aff1 = linear([self.x], 4, True, scope='aff1')
        relu1 = tf.nn.relu(aff1, name='relu1')
        aff2 = linear([relu1], 2, True, scope='aff2')
        yp = tf.nn.softmax(aff2, name='yp')
        self.logits = aff2
        self.yp = yp

    def _build_loss(self):
        ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.y, name='loss'))
        tf.add_to_collection('losses', ce_loss)
        self.loss = tf.add_n(tf.get_collection('losses'))
        tf.scalar_summary('loss', self.loss)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, supervised=True):
        assert isinstance(batch, DataSet)
        N, d = self.config.batch_size, self.config.dim
        feed_dict = {}

        x = np.zeros([N, d], dtype='float')
        feed_dict[self.x] = x
        X = batch.data['X']
        for i, xi in enumerate(X):
            for j, xij in enumerate(xi):
                x[i, j] = xij

        if supervised:
            y = np.zeros([N], dtype='int')
            feed_dict[self.y] = y
            Y = batch.data['Y']
            for i, yi in enumerate(Y):
                y[i] = yi

        return feed_dict


