import tensorflow as tf
from model.base_model import BaseTower
import numpy as np

from my.tensorflow.nn import linear


class Tower(BaseTower):
    def _initialize(self):
        params = self.params
        ph = self.placeholders
        tensors = self.tensors
        N = params.batch_size

        is_train = tf.placeholder('bool', shape=[], name='is_train')
        # TODO : define placeholders and put them in ph
        num_classes = params.num_classes
        x = tf.placeholder("float", shape=[N, 1], name='x')
        y = tf.placeholder("int32", shape=[N], name='y')
        ph['x'] = x
        ph['y'] = y
        ph['is_train'] = is_train

        # TODO : put your codes here
        with tf.variable_scope("main"):
            logits = linear([x], num_classes, True, scope='logits')

        with tf.name_scope("eval"):
            yp = tf.cast(tf.argmax(logits, 1), 'int32')
            correct = tf.equal(yp, y)
            # TODO : this must be properly defined
            tensors['correct'] = correct

        with tf.name_scope("loss"):
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y, name='ce')
            avg_ce = tf.reduce_mean(ce, name='avg_ce')
            tf.add_to_collection('losses', avg_ce)

            losses = tf.get_collection('losses')
            loss = tf.add_n(losses, name='loss')
            # TODO : this must be properly defined
            tensors['loss'] = loss

    def _get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        ph = self.placeholders
        N = params.batch_size
        # TODO : put more parameters

        # TODO : define your inputs to _initialize here
        x = np.zeros([N, 1], dtype='float')
        y = np.zeros([N], dtype='int32')
        feed_dict = {ph['x']: x, ph['y']: y,
                     ph['is_train']: mode == 'train'}

        # Batch can be empty in multi GPU parallelization
        if batch is None:
            return feed_dict

        # TODO : retrieve data and store it in the numpy arrays; example shown below
        X, Y = batch['X'], batch['Y']

        for i, xx in enumerate(X):
            x[i, 0] = xx
        for i, yy in enumerate(Y):
            y[i] = yy

        return feed_dict
