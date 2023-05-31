import random

import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from basic.read_data import DataSet
from my.tensorflow import get_initializer
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell


def get_multi_gpu_models(config):
    models = []
    for gpu_idx in range(config.num_gpus):
        with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
            model = Model(config, scope, rep=gpu_idx == 0)
            tf.get_variable_scope().reuse_variables()
            models.append(model)
    return models


class Model(object):
    def __init__(self, config, scope, rep=True):
        self.scope = scope
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here

        # N：batch size，即一批次训练/测试的样本数量。
        # M：最大段落数，即上下文文本最多可以有多少个段落。
        # JX：最大句子长度，即一个句子最多可以有多少个单词。
        # JQ：最大问题长度，即一个问题最多可以有多少个单词。
        # VW：单词词汇表的大小。
        # VC：字符词汇表的大小。
        # W：字符的最大长度。
        N, M, JX, JQ, VW, VC, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
        
        # self.x: 输入的文章内容，维度为[N, M, JX]，其中N为batch size，M为最大段落数，JX为最大段落长度。
        # self.cx: 输入的文章内容的字符级别的embedding，维度为[N, M, JX, W]，其中W为字符级别的embedding长度。
        # self.x_mask: 输入的文章内容的掩码，维度为[N, M, JX]，用于在self_attention中遮盖padding的段落内容。
        # self.q: 输入的问题内容，维度为[N, JQ]，其中JQ为最大问题长度。
        # self.cq: 输入的问题内容的字符级别的embedding，维度为[N, JQ, W]，其中W为字符级别的embedding长度。
        # self.q_mask: 输入的问题内容的掩码，维度为[N, JQ]，用于在self_attention中遮盖padding的问题内容。
        # self.y: 对答案的起始位置的one-hot表示，维度为[N, M, JX]。
        # self.y2: 对答案的结束位置的one-hot表示，维度为[N, M, JX]。
        # self.is_train: 是否为训练模式的标志位，用于区分训练和测试模式。
        # self.new_emb_mat: 如果需要对out-of-vocabulary (OOV)的词使用GloVe embeddings，那么需要提供一个额外的embedding matrix，维度为[OOV count, embedding dimension]。
        self.x = tf.placeholder('int32', [N, None, None], name='x')
        self.cx = tf.placeholder('int32', [N, None, None, W], name='cx')
        self.x_mask = tf.placeholder('bool', [N, None, None], name='x_mask')
        self.q = tf.placeholder('int32', [N, None], name='q')
        self.cq = tf.placeholder('int32', [N, None, W], name='cq')
        self.q_mask = tf.placeholder('bool', [N, None], name='q_mask')
        self.y = tf.placeholder('bool', [N, None, None], name='y')
        self.y2 = tf.placeholder('bool', [N, None, None], name='y2')
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
        self.var_ema = None
        if rep:
            self._build_var_ema()
        if config.mode == 'train':
            self._build_ema()

        self.summary = tf.merge_all_summaries()
        self.summary = tf.merge_summary(tf.get_collection("summaries", scope=self.scope))

    def _build_forward(self):
        config = self.config
        # N：batch size，即一批次训练/测试的样本数量。
        # M：最大段落数，即上下文文本最多可以有多少个段落。
        # JX：最大句子长度，即一个句子最多可以有多少个单词。
        # JQ：最大问题长度，即一个问题最多可以有多少个单词。
        # VW：单词词汇表的大小。
        # VC：字符词汇表的大小。
        # W：字符的最大长度。
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.max_word_size
        JX = tf.shape(self.x)[2]
        JQ = tf.shape(self.q)[1]
        M = tf.shape(self.x)[1]
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

        with tf.variable_scope("emb"):
            if config.use_char_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    # 定义了一个名为 char_emb_mat 的变量，表示字符级别的嵌入矩阵。该变量是一个可训练的 TensorFlow 变量，它的形状是 [VC, dc]，其中 VC 表示字符词汇表的大小，dc 表示字符嵌入的维度。
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

                with tf.variable_scope("char"):
                    Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, M, JX, W, dc]
                    Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]
                    Acx = tf.reshape(Acx, [-1, JX, W, dc])
                    Acq = tf.reshape(Acq, [-1, JQ, W, dc])

                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    heights = list(map(int, config.filter_heights.split(',')))
                    # (filter_sizes, dco)是一个元组，它被用作assert语句的第二个参数。如果assert语句失败，将会触发一个AssertionError异常，并输出这个元组的内容，以便于调试代码。在这里，如果sum(filter_sizes)不等于dco，则会输出filter_sizes和dco的值，帮助程序员定位问题。
                    assert sum(filter_sizes) == dco, (filter_sizes, dco)
                    with tf.variable_scope("conv"):
                        xx = multi_conv1d(Acx, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="xx")
                        if config.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
                        else:
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")
                        xx = tf.reshape(xx, [-1, M, JX, dco])
                        qq = tf.reshape(qq, [-1, JQ, dco])

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
                    self.tensor_dict['x'] = Ax
                    self.tensor_dict['q'] = Aq
                if config.use_char_emb:
                    xx = tf.concat(3, [xx, Ax])  # [N, M, JX, di]
                    qq = tf.concat(2, [qq, Aq])  # [N, JQ, di]
                else:
                    xx = Ax
                    qq = Aq

        # highway network
        if config.highway:
            with tf.variable_scope("highway"):
                xx = highway_network(xx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                qq = highway_network(qq, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)

        self.tensor_dict['xx'] = xx
        self.tensor_dict['qq'] = qq

        cell = BasicLSTMCell(d, state_is_tuple=True)
        # SwitchableDropoutWrapper 是继承自 DropoutWrapper 的一个类，它的作用是在 LSTM 层中添加可切换的 dropout。
        # 这个类的作用是在训练时执行 dropout，在测试时不执行 dropout，以避免过拟合。
        d_cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        # self.x_mask 是输入矩阵 self.x 的掩码矩阵，大小为 [N, M, JX]。其中 N 表示批量大小，M 表示文章中最大的句子数，JX 表示最长的句子长度。
        # 掩码矩阵中为 True 的位置表示输入中有单词，False 的位置则表示该位置是填充的。
        # 通过 tf.reduce_sum 函数对第三个维度进行求和，得到一个大小为 [N, M] 的矩阵 x_len，其中每个元素表示对应句子的长度（即实际上有多少单词是有效的）。
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("prepro"):
            # 实现双向LSTM
            # fw_u和bw_u是分别表示前向和后向LSTM输出的张量，shape为 [N, JQ, d]
            # fw_u_f和bw_u_f是分别表示前向和后向LSTM最终状态的元组，其中每个元素的shape为[N, d]
            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = bidirectional_dynamic_rnn(d_cell, d_cell, qq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]
            u = tf.concat(2, [fw_u, bw_u])
            if config.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
                h = tf.concat(3, [fw_h, bw_h])  # [N, M, JX, 2d]
            else:
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='h1')  # [N, M, JX, 2d]
                h = tf.concat(3, [fw_h, bw_h])  # [N, M, JX, 2d]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h

        with tf.variable_scope("main"):
            # h content, u query
            if config.dynamic_att:
                p0 = h
                u = tf.reshape(tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1]), [N * M, JQ, 2 * d])
                q_mask = tf.reshape(tf.tile(tf.expand_dims(self.q_mask, 1), [1, M, 1]), [N * M, JQ])
                first_cell = AttentionCell(cell, u, mask=q_mask, mapper='sim',
                                           input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
            else:
                p0 = attention_layer(config, self.is_train, h, u, h_mask=self.x_mask, u_mask=self.q_mask, scope="p0", tensor_dict=self.tensor_dict)
                first_cell = d_cell

            (fw_g0, bw_g0), _ = bidirectional_dynamic_rnn(first_cell, first_cell, p0, x_len, dtype='float', scope='g0')  # [N, M, JX, 2d]
            g0 = tf.concat(3, [fw_g0, bw_g0])
            (fw_g1, bw_g1), _ = bidirectional_dynamic_rnn(first_cell, first_cell, g0, x_len, dtype='float', scope='g1')  # [N, M, JX, 2d]
            g1 = tf.concat(3, [fw_g1, bw_g1])

            logits = get_logits([g1, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                mask=self.x_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')
            a1i = softsel(tf.reshape(g1, [N, M * JX, 2 * d]), tf.reshape(logits, [N, M * JX]))
            a1i = tf.tile(tf.expand_dims(tf.expand_dims(a1i, 1), 1), [1, M, JX, 1])

            (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(d_cell, d_cell, tf.concat(3, [p0, g1, a1i, g1 * a1i]),
                                                          x_len, dtype='float', scope='g2')  # [N, M, JX, 2d]
            g2 = tf.concat(3, [fw_g2, bw_g2])
            logits2 = get_logits([g2, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                 mask=self.x_mask,
                                 is_train=self.is_train, func=config.answer_func, scope='logits2')

            flat_logits = tf.reshape(logits, [-1, M * JX])
            flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
            yp = tf.reshape(flat_yp, [-1, M, JX])
            flat_logits2 = tf.reshape(logits2, [-1, M * JX])
            flat_yp2 = tf.nn.softmax(flat_logits2)
            yp2 = tf.reshape(flat_yp2, [-1, M, JX])

            self.tensor_dict['g1'] = g1
            self.tensor_dict['g2'] = g2

            self.logits = flat_logits
            self.logits2 = flat_logits2
            self.yp = yp
            self.yp2 = yp2

    def _build_loss(self):
        config = self.config
        JX = tf.shape(self.x)[2]
        M = tf.shape(self.x)[1]
        JQ = tf.shape(self.q)[1]
        loss_mask = tf.reduce_max(tf.cast(self.q_mask, 'float'), 1)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            self.logits, tf.cast(tf.reshape(self.y, [-1, M * JX]), 'float'))
        ce_loss = tf.reduce_mean(loss_mask * losses)
        tf.add_to_collection('losses', ce_loss)
        ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.logits2, tf.cast(tf.reshape(self.y2, [-1, M * JX]), 'float')))
        tf.add_to_collection("losses", ce_loss2)

        self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
        tf.scalar_summary(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    def _build_ema(self):
        self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema = self.ema
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
        ema_op = ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = ema.average(var)
            tf.scalar_summary(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = ema.average(var)
            tf.histogram_summary(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def _build_var_ema(self):
        self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
        ema = self.var_ema
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

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

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
                new_JX = 1
            else:
                new_JX = max(len(sent) for para in batch.data['x'] for sent in para)
            JX = min(JX, new_JX)

            if sum(len(ques) for ques in batch.data['q']) == 0:
                new_JQ = 1
            else:
                new_JQ = max(len(ques) for ques in batch.data['q'])
            JQ = min(JQ, new_JQ)

        if config.cpu_opt:
            if sum(len(para) for para in batch.data['x']) == 0:
                new_M = 1
            else:
                new_M = max(len(para) for para in batch.data['x'])
            M = min(M, new_M)

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
        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

        X = batch.data['x']
        CX = batch.data['cx']

        if supervised:
            y = np.zeros([N, M, JX], dtype='bool')
            y2 = np.zeros([N, M, JX], dtype='bool')
            feed_dict[self.y] = y
            feed_dict[self.y2] = y2

            for i, (xi, cxi, yi) in enumerate(zip(X, CX, batch.data['y'])):
                start_idx, stop_idx = random.choice(yi)
                j, k = start_idx
                j2, k2 = stop_idx
                if config.single:
                    X[i] = [xi[j]]
                    CX[i] = [cxi[j]]
                    j, j2 = 0, 0
                if config.squash:
                    offset = sum(map(len, xi[:j]))
                    j, k = 0, k + offset
                    offset = sum(map(len, xi[:j2]))
                    j2, k2 = 0, k2 + offset
                y[i, j, k] = True
                y2[i, j2, k2-1] = True

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
                q[i, j] = _get_word(qij)
                q_mask[i, j] = True

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        return feed_dict


def bi_attention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "bi_attention"):
        # h content, u query
        # h [N, M, JX, 2d]
        # u [N, JQ, 2d]
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        h_aug = tf.tile(tf.expand_dims(h, 3), [1, 1, 1, JQ, 1]) # [N, M, JX, JQ, 2d]
        u_aug = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1, 1]) # [N, M, JX, JQ, 2d]
        if h_mask is None:
            hu_mask = None
        else:
            h_mask_aug = tf.tile(tf.expand_dims(h_mask, 3), [1, 1, 1, JQ])
            u_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(u_mask, 1), 1), [1, M, JX, 1])
            # 按位取与（AND 操作），得到形状为 [N, M, JX, JQ] 的 hu_mask Tensor，表示 h 和 u 中每个位置是否都有效。
            hu_mask = h_mask_aug & u_mask_aug

        u_logits = get_logits([h_aug, u_aug], None, True, wd=config.wd, mask=hu_mask,
                              is_train=is_train, func=config.logit_func, scope='u_logits')  # [N, M, JX, JQ]
        # softsel是一个softmax加权的sum操作，可以理解为在target这个张量的最后一个维度（即d维度）上，
        # 对target的每一个元素乘以对应的logits的softmax值，然后对最后一个维度求和，最终得到一个新的张量。
        # 使用query中的词根据权重表示content中的词
        u_a = softsel(u_aug, u_logits)  # [N, M, JX, d]
        h_a = softsel(h, tf.reduce_max(u_logits, 3))  # [N, M, d]
        h_a = tf.tile(tf.expand_dims(h_a, 2), [1, 1, JX, 1])

        if tensor_dict is not None:
            a_u = tf.nn.softmax(u_logits)  # [N, M, JX, JQ]
            a_h = tf.nn.softmax(tf.reduce_max(u_logits, 3))
            tensor_dict['a_u'] = a_u
            tensor_dict['a_h'] = a_h
            variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=tf.get_variable_scope().name)
            for var in variables:
                tensor_dict[var.name] = var

        return u_a, h_a


def attention_layer(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "attention_layer"):
        # h content, u query
        # h [N, M, JX, 2d]
        # u [N, JQ, 2d]
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        if config.q2c_att or config.c2q_att:
            u_a, h_a = bi_attention(config, is_train, h, u, h_mask=h_mask, u_mask=u_mask, tensor_dict=tensor_dict)
        if not config.c2q_att:
            u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, M, JX, 1])
        if config.q2c_att:
            p0 = tf.concat(3, [h, u_a, h * u_a, h * h_a])
        else:
            p0 = tf.concat(3, [h, u_a, h * u_a])
        return p0