import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from my.tensorflow import get_initializer
from my.tensorflow.rnn_cell import DropoutWrapper

from cnn.base_model import BaseTower
import numpy as np

from my.tensorflow.nn import linear


def reverse_dynamic_rnn(cell, x, length, **kwargs):
    length = tf.cast(length, 'int64')
    x_r = tf.reverse_sequence(x, length, 1)
    out_r, state = dynamic_rnn(cell, x_r, length, **kwargs)
    out = tf.reverse_sequence(out_r, length, 1)
    return out, state


def conv1d(x, fh, ic, oc, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        f = tf.get_variable("filter", shape=[fh, 1, ic, oc], dtype='float')
        b = tf.get_variable("bias", shape=[oc], dtype='float')
        strides = [1, 1, 1, 1]
        out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, f, strides, "SAME"), b))
        return out


class Tower(BaseTower):
    def _initialize(self):
        params = self.params
        ph = self.placeholders
        tensors = self.tensors
        N = params.batch_size
        M = params.max_num_words
        K = params.max_ques_size
        char_vec_size = params.char_vec_size
        d = params.hidden_size
        V = params.vocab_size
        W = params.max_word_size
        C = params.char_vocab_size
        word_vec_size = params.word_vec_size
        filter_height = params.filter_height
        filter_stride = params.filter_stride
        num_layers = params.num_layers
        finetune = params.finetune

        is_train = tf.placeholder('bool', shape=[], name='is_train')
        # TODO : define placeholders and put them in ph
        x = tf.placeholder("int32", shape=[N, M], name='x')
        cx = tf.placeholder("int32", shape=[N, M, W], name='cx')
        q = tf.placeholder("int32", shape=[N, K], name='q')
        cq = tf.placeholder("int32", shape=[N, K, W], name='cq')
        y = tf.placeholder("int32", shape=[N], name='y')
        x_mask = tf.placeholder("bool", shape=[N, M], name='x_mask')
        q_mask = tf.placeholder("bool", shape=[N, K], name='q_mask')
        cx_mask = tf.placeholder("bool", shape=[N, M, W], name='cx_mask')
        cq_mask = tf.placeholder("bool", shape=[N, K, W], name='cq_mask')
        ph['x'] = x
        ph['cx'] = cx
        ph['q'] = q
        ph['cq'] = cq
        ph['y'] = y
        ph['x_mask'] = x_mask
        ph['cx_mask'] = cx_mask
        ph['q_mask'] = q_mask
        ph['cq_mask'] = cq_mask
        ph['is_train'] = is_train

        # TODO : put your codes here
        with tf.variable_scope("main") as vs:
            init_emb_mat = tf.constant(params.emb_mat, name='emb_mat')
            if finetune:
                emb_mat = tf.get_variable("emb_mat", shape=[V, word_vec_size], dtype='float', initializer=get_initializer(init_emb_mat))
            else:
                emb_mat = init_emb_mat
            char_emb_mat = tf.get_variable("char_emb_mat", shape=[C, char_vec_size], dtype='float')

            Ax = tf.nn.embedding_lookup(emb_mat, x, name='Ax')  # [N, M, w]
            Aq = tf.nn.embedding_lookup(emb_mat, q, name='Aq')  # [N, K, w]
            Acx = tf.nn.embedding_lookup(char_emb_mat, cx, name='Acx')  # [N, M, C, cd]
            Aqx = tf.nn.embedding_lookup(char_emb_mat, cq, name='Acq')  # [N, K, C, cd]
            Acx_adj = tf.reshape(Acx, [N*M, W, 1, char_vec_size])
            Aqx_adj = tf.reshape(Aqx, [N*K, W, 1, char_vec_size])
            with tf.variable_scope("char_emb"):
                Ax_c = tf.reshape(tf.reduce_max(conv1d(Acx_adj, filter_height, char_vec_size, d), 1), [N, M, d])
                tf.get_variable_scope().reuse_variables()
                Aq_c = tf.reshape(tf.reduce_max(conv1d(Aqx_adj, filter_height, char_vec_size, d), 1), [N, K, d])
            Ax = tf.concat(2, [Ax, Ax_c])
            Aq = tf.concat(2, [Aq, Aq_c])
            D = word_vec_size + d
            Ax_adj = tf.reshape(Ax, [N, M, 1, D])
            Aq_adj = tf.reshape(Aq, [N, K, 1, D])

            for layer_idx in range(num_layers):
                with tf.variable_scope("layer_{}".format(layer_idx)):
                    Ax_adj = conv1d(Ax_adj, filter_height, D, D, scope='Ax')
                    Aq_adj = conv1d(Aq_adj, filter_height, D, D, scope='Aq')

            Ax = tf.reshape(Ax_adj, [N, M, D])
            Aq_red = tf.reduce_mean(Aq_adj, 1)  # [N, 1, D]
            logits = linear(Ax * Aq_red, 1, True, squeeze=True)  # [N, M]
            VERY_BIG_NUMBER = 1e9
            logits += -VERY_BIG_NUMBER * tf.cast(tf.logical_not(x_mask), 'float')

        with tf.name_scope("loss"):
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y, name='ce')
            avg_ce = tf.reduce_mean(ce, name='avg_ce')
            tf.add_to_collection('losses', avg_ce)

            losses = tf.get_collection('losses')
            loss = tf.add_n(losses, name='loss')
            # TODO : this must be properly defined
            tensors['loss'] = loss

        with tf.name_scope("eval"):
            yp = tf.cast(tf.argmax(logits, 1), 'int32')
            correct = tf.equal(yp, y)
            # TODO : this must be properly defined
            tensors['correct'] = correct

    def _get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        ph = self.placeholders
        N = params.batch_size
        M = params.max_num_words
        K = params.max_ques_size
        W = params.max_word_size

        # TODO : put more parameters

        # TODO : define your inputs to _initialize here
        x = np.zeros([N, M], dtype='int32')
        cx = np.zeros([N, M, W], dtype='int32')
        q = np.zeros([N, K], dtype='int32')
        cq = np.zeros([N, K, W], dtype='int32')
        y = np.zeros([N], dtype='int32')
        x_mask = np.zeros([N, M], dtype='bool')
        cx_mask = np.zeros([N, M, W], dtype='bool')
        q_mask = np.zeros([N, K], dtype='bool')
        cq_mask = np.zeros([N, K, W], dtype='bool')

        feed_dict = {ph['x']: x, ph['q']: q, ph['y']: y,
                     ph['x_mask']: x_mask, ph['q_mask']: q_mask,
                     ph['cx']: cx, ph['cq']: cq, ph['cx_mask']: cx_mask, ph['cq_mask']: cq_mask,
                     ph['is_train']: mode == 'train'}

        # Batch can be empty in multi GPU parallelization
        if batch is None:
            return feed_dict

        X, Q, Y = batch['X'], batch['Q'], batch['Y']
        CX, CQ = batch['CX'], batch['CQ']
        for i, sents in enumerate(X):
            for k, word in enumerate(sents):
                x[i, k] = word
                x_mask[i, k] = True

        for i, ques in enumerate(Q):
            for j, word in enumerate(ques):
                q[i, j] = word
                q_mask[i, j] = True

        for i, idx in enumerate(Y):
            y[i] = idx

        for i, sents in enumerate(CX):
            for k, word in enumerate(sents):
                for l, char in enumerate(word):
                    cx[i, k, l] = char
                    cx_mask[i, k, l] = True

        for i, ques in enumerate(CQ):
            for j, word in enumerate(ques):
                for k, char in enumerate(word):
                    cq[i, j, k] = char
                    cq_mask[i, j, k] = True

        return feed_dict
