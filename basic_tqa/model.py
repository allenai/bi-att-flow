import random

import itertools
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell, GRUCell

from basic_tqa.read_data import DataSet
from basic_tqa.superhighway import SHCell
from my.tensorflow import exp_mask, get_initializer, VERY_SMALL_NUMBER
from my.tensorflow.nn import linear, double_linear_logits, linear_logits, softsel, dropout, get_logits, softmax, \
    highway_network, multi_conv1d
from my.tensorflow.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell
from my.utils import get_sim_idx


def attention_flow(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None, num_layers=1):
    """
    h_a:
    all u attending on h
    choosing an element of h that max-matches u
    First creates confusion matrix between h and u
    Then take max of the attention weights over u row
    Finally softmax over

    u_a:
    each h attending on u

    :param h: [N, M, JX, d]
    :param u: [N, JQ, d]
    :param h_mask:  [N, M, JX]
    :param u_mask:  [N, B]
    :param scope:
    :return: [N, M, d], [N, M, JX, d]
    """
    with tf.variable_scope(scope or "bi_attention"):
        d = config.hidden_size
        JX = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        h_aug = tf.tile(tf.expand_dims(h, 2), [1, 1, JQ, 1])
        u_aug = tf.tile(tf.expand_dims(u, 1), [1, JX, 1, 1])
        if h_mask is None:
            hu_mask = None
        else:
            h_mask_aug = tf.tile(tf.expand_dims(h_mask, 2), [1, 1, JQ])
            u_mask_aug = tf.tile(tf.expand_dims(u_mask, 1), [1, JX, 1])
            hu_mask = h_mask_aug & u_mask_aug

        u_logits = get_logits([h_aug, u_aug], None, True, wd=config.wd, mask=hu_mask,
                              is_train=is_train, func=config.logit_func, scope='u_logits')  # [N, JX, JQ]
        u_a = softsel(u_aug, u_logits)  # [N, JX, d]

        if tensor_dict is not None:
            a_u = tf.nn.softmax(u_logits)  # [N, JX, JQ]
            tensor_dict['a_u'] = a_u
            variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=tf.get_variable_scope().name)
            for var in variables:
                tensor_dict[var.name] = var

        p = tf.concat(2, [h, u_a, h * u_a])  # [N, JX, d]
        cell = BasicLSTMCell(d, state_is_tuple=True)
        d_cell = SwitchableDropoutWrapper(cell, is_train, input_keep_prob=config.input_keep_prob)
        h_len = tf.reduce_sum(tf.cast(h_mask, 'int32'), 1)
        g = p
        for layer_idx in range(num_layers):
            (fw_g, bw_g), _ = bidirectional_dynamic_rnn(d_cell, d_cell, g, h_len, dtype='float', scope='g{}'.format(layer_idx))  # [N*M, JX, 2d]
            g = tf.concat(2, [fw_g, bw_g])
        return p, g + h


def bi_attention_flow_layer(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "attention_layer"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        d = config.hidden_size
        h = tf.reshape(h, [-1, JX, 2*d])
        h_mask  = tf.reshape(h_mask, [-1, JX])
        u = tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1])  # [N, M, JQ, 2d]
        u = tf.reshape(u, [-1, JQ, 2*d])  # [N*M, JQ, 2d]
        u_mask = tf.tile(tf.expand_dims(u_mask, 1), [1, M, 1])
        u_mask = tf.reshape(u_mask, [-1, JQ])

        _, c2q = attention_flow(config, is_train, u, h, h_mask=u_mask, u_mask=h_mask, scope='c2q', num_layers=2)  # [N * M, JQ, 2d]
        p, q2c = attention_flow(config, is_train, h, c2q, h_mask=h_mask, u_mask=u_mask, scope='q2c', num_layers=2)  # [N * M, JX, 2d]
        p = tf.reshape(p, [-1, M, JX, 6*d])
        q2c = tf.reshape(q2c, [-1, M, JX, 2*d])
        out = p, q2c
        return out



class Model(object):
    def __init__(self, config, scope):
        self.scope = scope
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N, M, JX, JQ, VW, VC, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
        self.x = tf.placeholder('int32', [N, None, None], name='x')
        self.cx = tf.placeholder('int32', [N, None, None, W], name='cx')
        self.x_mask = tf.placeholder('bool', [N, None, None], name='x_mask')
        self.q = tf.placeholder('int32', [N, None], name='q')
        self.cq = tf.placeholder('int32', [N, None, W], name='cq')
        self.q_mask = tf.placeholder('bool', [N, None], name='q_mask')
        self.a = tf.placeholder('int32', [N, None, None], name='a')
        self.ca = tf.placeholder('int32', [N, None, None, W], name='ca')
        self.a_mask = tf.placeholder('bool', [N, None, None], name='a_mask')
        self.y = tf.placeholder('int32', [N], name='y')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')

        # Define misc
        self.tensor_dict = {}

        # Forward outputs / loss inputs
        self.logits = None
        self.yp = None
        self.var_list = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()
        if config.mode == 'train':
            self._build_ema()

        self.summary = tf.merge_all_summaries()
        self.summary = tf.merge_summary(tf.get_collection("summaries", scope=self.scope))

    def _build_forward(self):
        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.max_word_size
        JX = tf.shape(self.x)[2]
        JQ = tf.shape(self.q)[1]
        JA = tf.shape(self.a)[2]
        MA = tf.shape(self.a)[1]
        M = tf.shape(self.x)[1]
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

        with tf.variable_scope("emb"):
            if config.use_char_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

                with tf.variable_scope("char"):
                    Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, M, JX, W, dc]
                    Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]
                    Aca = tf.nn.embedding_lookup(char_emb_mat, self.ca)
                    Acx = tf.reshape(Acx, [-1, JX, W, dc])
                    Acq = tf.reshape(Acq, [-1, JQ, W, dc])
                    Aca = tf.reshape(Aca, [-1, JA, W, dc])

                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    heights = list(map(int, config.filter_heights.split(',')))
                    assert sum(filter_sizes) == dco
                    with tf.variable_scope("conv"):
                        xx = multi_conv1d(Acx, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="xx")
                        if config.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
                            tf.get_variable_scope().reuse_variables()
                            aa = multi_conv1d(Aca, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
                        else:
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")
                            aa = multi_conv1d(Aca, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="aa")
                        xx = tf.reshape(xx, [-1, M, JX, dco])
                        qq = tf.reshape(qq, [-1, JQ, dco])
                        aa = tf.reshape(aa, [-1, MA, JA, dco])

            if config.use_word_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    if config.mode == 'train':
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(config.emb_mat))
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
                    if config.use_glove_for_unk:
                        word_emb_mat = tf.concat(0, [word_emb_mat, self.new_emb_mat])

                with tf.name_scope("word"):
                    Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d]
                    Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]
                    Aa = tf.nn.embedding_lookup(word_emb_mat, self.a)  # [N, MA, JA, d]
                    self.tensor_dict['x'] = Ax
                    self.tensor_dict['q'] = Aq
                    self.tensor_dict['a'] = Aa
                if config.use_char_emb:
                    xx = tf.concat(3, [xx, Ax])  # [N, M, JX, di]
                    qq = tf.concat(2, [qq, Aq])  # [N, JQ, di]
                    aa = tf.concat(3, [aa, Aa])  # [N, MA, JA, di]
                else:
                    xx = Ax
                    qq = Aq
                    aa = Aa

        # highway network
        if config.highway:
            with tf.variable_scope("highway"):
                xx = highway_network(xx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                qq = highway_network(qq, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                aa = highway_network(aa, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)

        self.tensor_dict['xx'] = xx
        self.tensor_dict['qq'] = qq
        self.tensor_dict['aa'] = aa

        cell = BasicLSTMCell(d, state_is_tuple=True)
        d_cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]
        a_len = tf.reduce_sum(tf.cast(self.a_mask, 'int32'), 2)  # [N, MA]

        with tf.variable_scope("prepro"):
            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = bidirectional_dynamic_rnn(d_cell, d_cell, qq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]
            u = tf.concat(2, [fw_u, bw_u])
            if config.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
                h = tf.concat(3, [fw_h, bw_h])  # [N, M, JX, 2d]
                tf.get_variable_scope().reuse_variables()
                (fw_b, bw_b), _ = bidirectional_dynamic_rnn(cell, cell, aa, a_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
                b = tf.concat(3, [fw_b, bw_b])  # [N, MA, JA, 2d]

            else:
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='h1')  # [N, M, JX, 2d]
                h = tf.concat(3, [fw_h, bw_h])  # [N, M, JX, 2d]
                (fw_b, bw_b), _ = bidirectional_dynamic_rnn(cell, cell, aa, a_len, dtype='float', scope='b1')  # [N, M, JX, 2d]
                b = tf.concat(3, [fw_b, bw_b])  # [N, MA, JA, 2d]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h
            self.tensor_dict['b'] = b

        with tf.variable_scope("main"):
            p, q2c = bi_attention_flow_layer(config, self.is_train, h, u, h_mask=self.x_mask, u_mask=self.q_mask)
            att_logits = get_logits([p, q2c], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                mask=self.x_mask, is_train=self.is_train, func=config.answer_func, scope='att_logits')
            ap = softsel(tf.reshape(h, [N, M * JX, 2 * d]), tf.reshape(att_logits, [N, M * JX]))  # [N, 2d]
            ap_tiled = tf.tile(tf.expand_dims(ap, 1), [1, MA, 1])
            b_sum = tf.reduce_sum(b, 2)  # [N, MA, 2d]
            logits = get_logits([ap_tiled, b_sum, ap_tiled * b_sum], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                mask=tf.reduce_any(self.a_mask, 2), is_train=self.is_train, func=config.answer_func, scope='logits')  # [N, MA]
            self.logits = logits
            self.yp = tf.nn.softmax(exp_mask(self.logits, tf.reduce_any(self.a_mask, 2)))

    def _build_loss(self):
        config = self.config
        N, M, JX, JQ, VW, VC = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size
        JX = tf.shape(self.x)[2]
        M = tf.shape(self.x)[1]
        JQ = tf.shape(self.q)[1]
        loss_mask = tf.reduce_max(tf.cast(self.q_mask, 'float'), 1)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.y)
        ce_loss = tf.reduce_mean(loss_mask * losses)
        tf.add_to_collection('losses', ce_loss)

        self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
        tf.scalar_summary(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    def _build_ema(self):
        ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema_op = ema.apply(tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/histogram", scope=self.scope))
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = ema.average(var)
            tf.scalar_summary(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/histogram", scope=self.scope):
            ema_var = ema.average(var)
            tf.histogram_summary(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        # FIXME : tf-idf
        X = []
        CX = []
        for Xi, CXi, Qi in zip(batch.data['x'], batch.data['cx'], batch.data['q']):
            counters = [Counter(word.lower() for sent in para for word in sent) for para in Xi]
            question = [word.lower() for word in Qi]
            idx = get_sim_idx(question, counters)
            X.append(Xi[idx])
            CX.append(CXi[idx])
            """
            print(len(Xi))
            print("all:", "\n\n".join(" ".join(" ".join(words) for words in sents) for sents in Xi))
            print("question:", " ".join(Qi))
            print("para:", " ".join(" ".join(each) for each in Xi[idx]))
            """
        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        JA, MA = config.max_ans_size, config.max_num_anss
        feed_dict = {}

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for para in X for sent in para) == 0:
                new_JX = 1
            else:
                new_JX = max(len(sent) for para in X for sent in para)
            JX = min(JX, new_JX)

            if sum(len(ques) for ques in batch.data['q']) == 0:
                new_JQ = 1
            else:
                new_JQ = max(len(ques) for ques in batch.data['q'])
            JQ = min(JQ, new_JQ)

            if sum(len(ans) for anss in batch.data['a'] for ans in anss) == 0:
                new_JA = 1
            else:
                new_JA = max(len(ans) for anss in batch.data['a'] for ans in anss)
            JA = min(JA, new_JA)

        if config.cpu_opt:
            if sum(len(para) for para in X) == 0:
                new_M = 1
            else:
                new_M = max(len(para) for para in X)
            M = min(M, new_M)

            if sum(len(anss) for anss in batch.data['a']) == 0:
                new_MA = 1
            else:
                new_MA = max(len(anss) for anss in batch.data['a'])
            MA = min(MA, new_MA)

        x = np.zeros([N, M, JX], dtype='int32')
        cx = np.zeros([N, M, JX, W], dtype='int32')
        x_mask = np.zeros([N, M, JX], dtype='bool')
        q = np.zeros([N, JQ], dtype='int32')
        cq = np.zeros([N, JQ, W], dtype='int32')
        q_mask = np.zeros([N, JQ], dtype='bool')
        a = np.zeros([N, MA, JA], dtype='int32')
        ca = np.zeros([N, MA, JA, W], dtype='int32')
        a_mask = np.zeros([N, MA, JA], dtype='bool')

        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        feed_dict[self.cq] = cq
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.a] = a
        feed_dict[self.ca] = ca
        feed_dict[self.a_mask] = a_mask
        feed_dict[self.is_train] = is_train
        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']


        if supervised:
            y = np.zeros([N], dtype='int32')
            feed_dict[self.y] = y
            # feed_dict[self.yy] = yy

            for i, yi in enumerate(batch.data['y']):
                y[i] = yi

        def _get_word(word):
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            if config.use_glove_for_unk:
                d2 = batch.shared['new_word2idx']
                for each in (word, word.lower(), word.capitalize(), word.upper()):
                    if each in d2:
                        return d2[each] + len(d)
            return 1

        def _get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1

        for i, xi in enumerate(X):
            if self.config.squash:
                xi = [list(itertools.chain(*xi))]
            for j, xij in enumerate(xi):
                if j == config.max_num_sents:
                    break
                for k, xijk in enumerate(xij):
                    if k == config.max_sent_size:
                        break
                    each = _get_word(xijk)
                    assert isinstance(each, int), each
                    x[i, j, k] = each
                    x_mask[i, j, k] = True

        for i, cxi in enumerate(CX):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
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
                if j == config.max_ques_size:
                    break
                q[i, j] = _get_word(qij)
                q_mask[i, j] = True

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                if j == config.max_ques_size:
                    break
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        for i, ai in enumerate(batch.data['a']):
            for j, aij in enumerate(ai):
                for k, aijk in enumerate(aij):
                    a[i, j, k] = _get_word(aijk)
                    a_mask[i, j, k] = True

        for i, cai in enumerate(batch.data['ca']):
            for j, caij in enumerate(cai):
                for k, caijk in enumerate(caij):
                    for l, caijkl in enumerate(caijk):
                        ca[i, j, k, l] = _get_char(caijkl)
                        if l + 1 == config.max_word_size:
                            break

        return feed_dict


def get_multi_gpu_models(config):
    models = []
    for gpu_idx in range(config.num_gpus):
        with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
            model = Model(config, scope)
            tf.get_variable_scope().reuse_variables()
            models.append(model)
    return models
