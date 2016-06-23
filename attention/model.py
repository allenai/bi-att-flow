import tensorflow as tf
from attention.base_model import BaseTower
import numpy as np


class Tower(BaseTower):
    def _initialize(self):
        params = self.params
        ph = self.placeholders
        tensors = self.tensors
        N = params.batch_size

        # TODO : define placeholders and put them in ph
        x = tf.placeholder("int32", shape=[N], name='x')
        y = tf.placeholder("bool", shape=[N], name='y')
        ph['x'] = x
        ph['y'] = y

        # TODO : put your codes here

        with tf.name_scope("eval"):
            # TODO : this must be properly defined
            tensors['correct'] = None

        with tf.name_scope("loss"):
            # TODO : this must be properly defined
            tensors['loss'] = None

    def _get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        ph = self.placeholders
        N = params.batch_size
        # TODO : put more parameters

        # TODO : define your inputs to _initialize here
        x = np.zeros([N], dtype='int32')
        y = np.zeros([N], dtype='bool')
        feed_dict = {ph['x']: x, ph['y']: y,
                     ph['is_train']: mode == 'train'}

        if batch is None:
            return feed_dict

        for i, xx in enumerate(x):
            x[i] = xx
        for i, yy in enumerate(y):
            y[i] = yy

        return feed_dict
