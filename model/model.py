import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from my.tensorflow import get_initializer
from my.tensorflow.rnn_cell import DropoutWrapper

from model.base_model import BaseTower
import numpy as np

from my.tensorflow.nn import linear


def reverse_dynamic_rnn(cell, x, length, **kwargs):
    length = tf.cast(length, 'int64')
    x_r = tf.reverse_sequence(x, length, 1)
    out_r, state = dynamic_rnn(cell, x_r, length, **kwargs)
    out = tf.reverse_sequence(out_r, length, 1)
    return out, state


class Tower(BaseTower):
    def _initialize(self):
        params = self.params
        ph = self.placeholders
        tensors = self.tensors
        N = params.batch_size
        M = params.max_num_sents
        J = params.max_sent_size
        K = params.max_ques_size
        d = params.hidden_size
        V = params.vocab_size
        W = params.max_word_size
        C = params.char_vocab_size
        word_vec_size = params.word_vec_size
        filter_height = params.filter_height
        filter_stride = params.filter_stride
        keep_prob = params.keep_prob
        finetune = params.finetune

        is_train = tf.placeholder('bool', shape=[], name='is_train')
        # TODO : define placeholders and put them in ph
        x = tf.placeholder("int32", shape=[N, M, J], name='x')
        cx = tf.placeholder("int32", shape=[N, M, J, W], name='cx')
        q = tf.placeholder("int32", shape=[N, K], name='q')
        cq = tf.placeholder("int32", shape=[N, K, W], name='cq')
        y = tf.placeholder("int32", shape=[N, 2], name='y')
        x_mask = tf.placeholder("bool", shape=[N, M, J], name='x_mask')
        q_mask = tf.placeholder("bool", shape=[N, K], name='q_mask')
        cx_mask = tf.placeholder("bool", shape=[N, M, J, W], name='cx_mask')
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
            Ax = tf.nn.embedding_lookup(emb_mat, x, name='Ax')  # [N, M, J, w]
            Aq = tf.nn.embedding_lookup(emb_mat, q, name='Aq')  # [N, K, w]

            char_emb_mat = tf.get_variable("char_emb_mat", shape=[C, d], dtype='float')
            Acx = tf.nn.embedding_lookup(char_emb_mat, cx, name='Acx')  # [N, M, J, C, d]
            Aqx = tf.nn.embedding_lookup(char_emb_mat, cq, name='Acq')  # [N, K, C, d]
            Acx_adj = tf.reshape(Acx, [N*M*J, W, 1, d])
            Aqx_adj = tf.reshape(Aqx, [N*K, W, 1, d])
            filter = tf.get_variable("filter", shape=[filter_height, 1, d, d], dtype='float')
            strides = [1, filter_stride, 1, 1]
            Acx_conv = tf.nn.conv2d(Acx_adj, filter, strides, "VALID")  # [N*M*J, C/filter_stride, 1, d]
            Aqx_conv = tf.nn.conv2d(Aqx_adj, filter, strides, "VALID")  # [N*K, C/filter_stride, 1, d]
            Ax_c = tf.reshape(tf.reduce_max(tf.nn.relu(Acx_conv), 1), [N, M, J, d])
            Aq_c = tf.reshape(tf.reduce_max(tf.nn.relu(Aqx_conv), 1), [N, K, d])

            Ax = tf.concat(3, [Ax, Ax_c])  # [N, M, J, w+d]
            Aq = tf.concat(2, [Aq, Aq_c])  # [N, K, w+d]



            q_length = tf.reduce_sum(tf.cast(q_mask, 'int32'), 1)  # [N]
            D = word_vec_size + d
            cell = BasicLSTMCell(D, state_is_tuple=True)
            cell = DropoutWrapper(cell, input_keep_prob=keep_prob, is_train=is_train)
            Ax_flat = tf.reshape(Ax, [N*M, J, D])
            x_sent_length = tf.reduce_sum(tf.cast(tf.reshape(x_mask, [N*M, J]), 'int32'), 1)  # [N*M]
            Ax_flat_out_fw, _ = dynamic_rnn(cell, Ax_flat, x_sent_length, dtype='float', scope='fw')  # [N*M, J, d]
            Ax_flat_out_bw, _ = reverse_dynamic_rnn(cell, Ax_flat, x_sent_length, dtype='float', scope='bw')
            Ax_flat_out_fw = tf.reshape(Ax_flat_out_fw, [N, M*J, D])
            Ax_flat_out_bw = tf.reshape(Ax_flat_out_bw, [N, M*J, D])
            vs.reuse_variables()
            _, (_, Aq_final_fw) = dynamic_rnn(cell, Aq, q_length, dtype='float', scope='fw')  # [N, d]
            _, (_, Aq_final_bw) = reverse_dynamic_rnn(cell, Aq, q_length, dtype='float', scope='bw')  # [N, d]
            Ax_flat_out = tf.concat(2, [Ax_flat_out_fw, Ax_flat_out_bw])
            Aq_final = tf.concat(1, [Aq_final_fw, Aq_final_bw])
            Aq_final_aug = tf.expand_dims(Aq_final, 1)  # [N, 1,  d]

        with tf.variable_scope("logit"):
            logits_flat = linear(Ax_flat_out * Aq_final_aug, 1, True, squeeze=True)  # [N, M*J]

        with tf.name_scope("loss"):
            y_flat = tf.reduce_sum(y * tf.constant([J, 1]), 1)  # [N]
            x_mask_flat = tf.reshape(x_mask, [N, M*J])
            VERY_BIG_NUMBER = 1e9
            logits_flat += -VERY_BIG_NUMBER * tf.cast(tf.logical_not(x_mask_flat), 'float')
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, y_flat, name='ce')
            avg_ce = tf.reduce_mean(ce, name='avg_ce')
            tf.add_to_collection('losses', avg_ce)

            losses = tf.get_collection('losses')
            loss = tf.add_n(losses, name='loss')
            # TODO : this must be properly defined
            tensors['loss'] = loss

        with tf.name_scope("eval"):
            yp_flat = tf.cast(tf.argmax(logits_flat, 1), 'int32')
            correct = tf.equal(yp_flat, y_flat)
            # TODO : this must be properly defined
            tensors['correct'] = correct


    def _get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        ph = self.placeholders
        N = params.batch_size
        M = params.max_num_sents
        J = params.max_sent_size
        K = params.max_ques_size
        W = params.max_word_size

        # TODO : put more parameters

        # TODO : define your inputs to _initialize here
        x = np.zeros([N, M, J], dtype='int32')
        cx = np.zeros([N, M, J, W], dtype='int32')
        q = np.zeros([N, K], dtype='int32')
        cq = np.zeros([N, K, W], dtype='int32')
        y = np.zeros([N, 2], dtype='int32')
        x_mask = np.zeros([N, M, J], dtype='bool')
        cx_mask = np.zeros([N, M, J, W], dtype='bool')
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
            for j, sent in enumerate(sents):
                for k, word in enumerate(sent):
                    x[i, j, k] = word
                    x_mask[i, j, k] = True

        for i, ques in enumerate(Q):
            for j, word in enumerate(ques):
                q[i, j] = word
                q_mask[i, j] = True

        for i, idxs in enumerate(Y):
            for j, idx in enumerate(idxs):
                y[i, j] = idx

        for i, sents in enumerate(CX):
            for j, sent in enumerate(sents):
                for k, word in enumerate(sent):
                    for l, char in enumerate(word):
                        cx[i, j, k, l] = char
                        cx_mask[i, j, k, l] = True

        for i, ques in enumerate(CQ):
            for j, word in enumerate(ques):
                for k, char in enumerate(word):
                    cq[i, j, k] = char
                    cq_mask[i, j, k] = True

        return feed_dict
