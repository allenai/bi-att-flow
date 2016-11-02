import itertools
from collections import defaultdict

import numpy as np
import tensorflow as tf
import os

from basic_cnn.read_data import DataSet
from my.nltk_utils import span_f1
from my.tensorflow import padded_reshape
from my.utils import argmax


class Evaluation(object):
    def __init__(self, data_type, global_step, idxs, yp, tensor_dict=None):
        self.data_type = data_type
        self.global_step = global_step
        self.idxs = idxs
        self.yp = yp
        self.num_examples = len(yp)
        self.tensor_dict = None
        self.dict = {'data_type': data_type,
                     'global_step': global_step,
                     'yp': yp,
                     'idxs': idxs,
                     'num_examples': self.num_examples}
        if tensor_dict is not None:
            self.tensor_dict = {key: val.tolist() for key, val in tensor_dict.items()}
            for key, val in self.tensor_dict.items():
                self.dict[key] = val
        self.summaries = None

    def __repr__(self):
        return "{} step {}".format(self.data_type, self.global_step)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        new_idxs = self.idxs + other.idxs
        new_tensor_dict = None
        if self.tensor_dict is not None:
            new_tensor_dict = {key: val + other.tensor_dict[key] for key, val in self.tensor_dict.items()}
        return Evaluation(self.data_type, self.global_step, new_idxs, new_yp, tensor_dict=new_tensor_dict)

    def __radd__(self, other):
        return self.__add__(other)


class LabeledEvaluation(Evaluation):
    def __init__(self, data_type, global_step, idxs, yp, y, id2answer_dict, tensor_dict=None):
        super(LabeledEvaluation, self).__init__(data_type, global_step, idxs, yp, tensor_dict=tensor_dict)
        self.y = y
        self.dict['y'] = y
        self.id2answer_dict = id2answer_dict

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        new_y = self.y + other.y
        new_idxs = self.idxs + other.idxs
        new_id2answer_dict = dict(list(self.id2answer_dict.items()) + list(other.id2answer_dict.items()))
        new_id2score_dict = dict(list(self.id2answer_dict['scores'].items()) + list(other.id2answer_dict['scores'].items()))
        new_id2answer_dict['scores'] = new_id2score_dict
        if self.tensor_dict is not None:
            new_tensor_dict = {key: np.concatenate((val, other.tensor_dict[key]), axis=0) for key, val in self.tensor_dict.items()}
        return LabeledEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_y, new_id2answer_dict, tensor_dict=new_tensor_dict)


class AccuracyEvaluation(LabeledEvaluation):
    def __init__(self, data_type, global_step, idxs, yp, y, id2answer_dict, correct, loss, tensor_dict=None):
        super(AccuracyEvaluation, self).__init__(data_type, global_step, idxs, yp, y, id2answer_dict, tensor_dict=tensor_dict)
        self.loss = loss
        self.correct = correct
        self.id2answer_dict = id2answer_dict
        self.acc = sum(correct) / len(correct)
        self.dict['loss'] = loss
        self.dict['correct'] = correct
        self.dict['acc'] = self.acc
        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=self.loss)])
        acc_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/acc'.format(data_type), simple_value=self.acc)])
        self.summaries = [loss_summary, acc_summary]

    def __repr__(self):
        return "{} step {}: accuracy={}={}/{}, loss={}".format(self.data_type, self.global_step, self.acc,
                                                               sum(self.correct), self.num_examples, self.loss)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_idxs = self.idxs + other.idxs
        new_yp = self.yp + other.yp
        new_y = self.y + other.y
        new_correct = self.correct + other.correct
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_correct)
        new_id2answer_dict = dict(list(self.id2answer_dict.items()) + list(other.id2answer_dict.items()))
        new_id2score_dict = dict(list(self.id2answer_dict['scores'].items()) + list(other.id2answer_dict['scores'].items()))
        new_id2answer_dict['scores'] = new_id2score_dict
        new_tensor_dict = None
        if self.tensor_dict is not None:
            new_tensor_dict = {key: np.concatenate((val, other.tensor_dict[key]), axis=0) for key, val in self.tensor_dict.items()}
        return AccuracyEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_y, new_id2answer_dict, new_correct, new_loss, tensor_dict=new_tensor_dict)


class Evaluator(object):
    def __init__(self, config, model, tensor_dict=None):
        self.config = config
        self.model = model
        self.global_step = model.global_step
        self.yp = model.yp
        self.tensor_dict = {} if tensor_dict is None else tensor_dict

    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        feed_dict = self.model.get_feed_dict(data_set, False, supervised=False)
        global_step, yp, vals = sess.run([self.global_step, self.yp, list(self.tensor_dict.values())], feed_dict=feed_dict)
        yp = yp[:data_set.num_examples]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = Evaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), tensor_dict=tensor_dict)
        return e

    def get_evaluation_from_batches(self, sess, batches):
        e = sum(self.get_evaluation(sess, batch) for batch in batches)
        return e


class LabeledEvaluator(Evaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(LabeledEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.y = model.y

    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        feed_dict = self.model.get_feed_dict(data_set, False, supervised=False)
        global_step, yp, vals = sess.run([self.global_step, self.yp, list(self.tensor_dict.values())], feed_dict=feed_dict)
        yp = yp[:data_set.num_examples]
        y = feed_dict[self.y]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = LabeledEvaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), y.tolist(), tensor_dict=tensor_dict)
        return e


class AccuracyEvaluator(LabeledEvaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(AccuracyEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.loss = model.loss

    def get_evaluation(self, sess, batch):
        idxs, data_set = self._split_batch(batch)
        assert isinstance(data_set, DataSet)
        feed_dict = self._get_feed_dict(batch)
        y = data_set.data['y']
        global_step, yp, loss, vals = sess.run([self.global_step, self.yp, self.loss, list(self.tensor_dict.values())], feed_dict=feed_dict)
        yp = yp[:data_set.num_examples]
        correct, probs, preds = zip(*[self.__class__.compare(data_set.get_one(idx), ypi) for idx, ypi in zip(data_set.valid_idxs, yp)])
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        ids = data_set.data['ids']
        id2score_dict = {id_: prob for id_, prob in zip(ids, probs)}
        id2answer_dict = {id_: pred for id_, pred in zip(ids, preds)}
        id2answer_dict['scores'] = id2score_dict
        e = AccuracyEvaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), y, id2answer_dict, correct, float(loss), tensor_dict=tensor_dict)
        return e

    @staticmethod
    def compare(data, ypi):
        prob = float(np.max(ypi))
        yi = data['y']
        for start, stop in yi:
            if start == int(np.argmax(ypi)):
                return True, prob, " "
        return False, prob, " "

    def _split_batch(self, batch):
        return batch

    def _get_feed_dict(self, batch):
        return self.model.get_feed_dict(batch[1], False)


class CNNAccuracyEvaluator(AccuracyEvaluator):
    @staticmethod
    def compare(data, ypi):
        # ypi: [N, M, JX] numbers
        yi = data['y'][0]  # entity
        xi = data['x'][0]  # [N, M, JX] words
        dist = defaultdict(int)
        for ypij, xij in zip(ypi, xi):
            for ypijk, xijk in zip(ypij, xij):
                if xijk.startswith("@"):
                    dist[xijk] += ypijk
        pred, prob = max(dist.items(), key=lambda item: item[1])
        assert pred.startswith("@")
        assert yi.startswith("@")
        return pred == yi, prob, pred


class AccuracyEvaluator2(AccuracyEvaluator):
    @staticmethod
    def compare(yi, ypi):
        for start, stop in yi:
            para_start = int(np.argmax(np.max(ypi, 1)))
            sent_start = int(np.argmax(ypi[para_start]))
            if tuple(start) == (para_start, sent_start):
                return True
        return False


class ForwardEvaluation(Evaluation):
    def __init__(self, data_type, global_step, idxs, yp, yp2, loss, id2answer_dict, tensor_dict=None):
        super(ForwardEvaluation, self).__init__(data_type, global_step, idxs, yp, tensor_dict=tensor_dict)
        self.yp2 = yp2
        self.loss = loss
        self.dict['loss'] = loss
        self.dict['yp2'] = yp2
        self.id2answer_dict = id2answer_dict

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_idxs = self.idxs + other.idxs
        new_yp = self.yp + other.yp
        new_yp2 = self.yp2 + other.yp2
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_yp)
        new_id2answer_dict = dict(list(self.id2answer_dict.items()) + list(other.id2answer_dict.items()))
        if self.tensor_dict is not None:
            new_tensor_dict = {key: np.concatenate((val, other.tensor_dict[key]), axis=0) for key, val in self.tensor_dict.items()}
        return ForwardEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_yp2, new_loss, new_id2answer_dict, tensor_dict=new_tensor_dict)

    def __repr__(self):
        return "{} step {}: loss={:.4f}".format(self.data_type, self.global_step, self.loss)


class F1Evaluation(AccuracyEvaluation):
    def __init__(self, data_type, global_step, idxs, yp, yp2, y, correct, loss, f1s, id2answer_dict, tensor_dict=None):
        super(F1Evaluation, self).__init__(data_type, global_step, idxs, yp, y, correct, loss, tensor_dict=tensor_dict)
        self.yp2 = yp2
        self.f1s = f1s
        self.f1 = float(np.mean(f1s))
        self.dict['yp2'] = yp2
        self.dict['f1s'] = f1s
        self.dict['f1'] = self.f1
        self.id2answer_dict = id2answer_dict
        f1_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/f1'.format(data_type), simple_value=self.f1)])
        self.summaries.append(f1_summary)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_idxs = self.idxs + other.idxs
        new_yp = self.yp + other.yp
        new_yp2 = self.yp2 + other.yp2
        new_y = self.y + other.y
        new_correct = self.correct + other.correct
        new_f1s = self.f1s + other.f1s
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_correct)
        new_id2answer_dict = dict(list(self.id2answer_dict.items()) + list(other.id2answer_dict.items()))
        return F1Evaluation(self.data_type, self.global_step, new_idxs, new_yp, new_yp2, new_y, new_correct, new_loss, new_f1s, new_id2answer_dict)

    def __repr__(self):
        return "{} step {}: accuracy={:.4f}, f1={:.4f}, loss={:.4f}".format(self.data_type, self.global_step, self.acc, self.f1, self.loss)


class F1Evaluator(LabeledEvaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(F1Evaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.yp2 = model.yp2
        self.loss = model.loss

    def get_evaluation(self, sess, batch):
        idxs, data_set = self._split_batch(batch)
        assert isinstance(data_set, DataSet)
        feed_dict = self._get_feed_dict(batch)
        global_step, yp, yp2, loss, vals = sess.run([self.global_step, self.yp, self.yp2, self.loss, list(self.tensor_dict.values())], feed_dict=feed_dict)
        y = data_set.data['y']
        if self.config.squash:
            new_y = []
            for xi, yi in zip(data_set.data['x'], y):
                new_yi = []
                for start, stop in yi:
                    start_offset = sum(map(len, xi[:start[0]]))
                    stop_offset = sum(map(len, xi[:stop[0]]))
                    new_start = 0, start_offset + start[1]
                    new_stop = 0, stop_offset + stop[1]
                    new_yi.append((new_start, new_stop))
                new_y.append(new_yi)
            y = new_y
        if self.config.single:
            new_y = []
            for yi in y:
                new_yi = []
                for start, stop in yi:
                    new_start = 0, start[1]
                    new_stop = 0, stop[1]
                    new_yi.append((new_start, new_stop))
                new_y.append(new_yi)
            y = new_y

        yp, yp2 = yp[:data_set.num_examples], yp2[:data_set.num_examples]
        spans = [get_best_span(ypi, yp2i) for ypi, yp2i in zip(yp, yp2)]

        def _get(xi, span):
            if len(xi) <= span[0][0]:
                return [""]
            if len(xi[span[0][0]]) <= span[1][1]:
                return [""]
            return xi[span[0][0]][span[0][1]:span[1][1]]

        id2answer_dict = {id_: " ".join(_get(xi, span))
                          for id_, xi, span in zip(data_set.data['ids'], data_set.data['x'], spans)}
        correct = [self.__class__.compare2(yi, span) for yi, span in zip(y, spans)]
        f1s = [self.__class__.span_f1(yi, span) for yi, span in zip(y, spans)]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = F1Evaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), yp2.tolist(), y,
                         correct, float(loss), f1s, id2answer_dict, tensor_dict=tensor_dict)
        return e

    def _split_batch(self, batch):
        return batch

    def _get_feed_dict(self, batch):
        return self.model.get_feed_dict(batch[1], False)

    @staticmethod
    def compare(yi, ypi, yp2i):
        for start, stop in yi:
            aypi = argmax(ypi)
            mask = np.zeros(yp2i.shape)
            mask[aypi[0], aypi[1]:] = np.ones([yp2i.shape[1] - aypi[1]])
            if tuple(start) == aypi and (stop[0], stop[1]-1) == argmax(yp2i * mask):
                return True
        return False

    @staticmethod
    def compare2(yi, span):
        for start, stop in yi:
            if tuple(start) == span[0] and tuple(stop) == span[1]:
                return True
        return False

    @staticmethod
    def span_f1(yi, span):
        max_f1 = 0
        for start, stop in yi:
            if start[0] == span[0][0]:
                true_span = start[1], stop[1]
                pred_span = span[0][1], span[1][1]
                f1 = span_f1(true_span, pred_span)
                max_f1 = max(f1, max_f1)
        return max_f1


class MultiGPUF1Evaluator(F1Evaluator):
    def __init__(self, config, models, tensor_dict=None):
        super(MultiGPUF1Evaluator, self).__init__(config, models[0], tensor_dict=tensor_dict)
        self.models = models
        with tf.name_scope("eval_concat"):
            N, M, JX = config.batch_size, config.max_num_sents, config.max_sent_size
            self.yp = tf.concat(0, [padded_reshape(model.yp, [N, M, JX]) for model in models])
            self.yp2 = tf.concat(0, [padded_reshape(model.yp2, [N, M, JX]) for model in models])
            self.loss = tf.add_n([model.loss for model in models])/len(models)

    def _split_batch(self, batches):
        idxs_list, data_sets = zip(*batches)
        idxs = sum(idxs_list, ())
        data_set = sum(data_sets, data_sets[0].get_empty())
        return idxs, data_set

    def _get_feed_dict(self, batches):
        feed_dict = {}
        for model, (_, data_set) in zip(self.models, batches):
            feed_dict.update(model.get_feed_dict(data_set, False))
        return feed_dict


class MultiGPUCNNAccuracyEvaluator(CNNAccuracyEvaluator):
    def __init__(self, config, models, tensor_dict=None):
        super(MultiGPUCNNAccuracyEvaluator, self).__init__(config, models[0], tensor_dict=tensor_dict)
        self.models = models
        with tf.name_scope("eval_concat"):
            N, M, JX = config.batch_size, config.max_num_sents, config.max_sent_size
            self.yp = tf.concat(0, [padded_reshape(model.yp, [N, M, JX]) for model in models])
            self.loss = tf.add_n([model.loss for model in models])/len(models)

    def _split_batch(self, batches):
        idxs_list, data_sets = zip(*batches)
        idxs = sum(idxs_list, ())
        data_set = sum(data_sets, data_sets[0].get_empty())
        return idxs, data_set

    def _get_feed_dict(self, batches):
        feed_dict = {}
        for model, (_, data_set) in zip(self.models, batches):
            feed_dict.update(model.get_feed_dict(data_set, False))
        return feed_dict


class ForwardEvaluator(Evaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(ForwardEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.yp2 = model.yp2
        self.loss = model.loss

    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)
        global_step, yp, yp2, loss, vals = sess.run([self.global_step, self.yp, self.yp2, self.loss, list(self.tensor_dict.values())], feed_dict=feed_dict)

        yp, yp2 = yp[:data_set.num_examples], yp2[:data_set.num_examples]
        spans = [get_best_span(ypi, yp2i) for ypi, yp2i in zip(yp, yp2)]

        def _get(xi, span):
            if len(xi) <= span[0][0]:
                return [""]
            if len(xi[span[0][0]]) <= span[1][1]:
                return [""]
            return xi[span[0][0]][span[0][1]:span[1][1]]

        id2answer_dict = {id_: " ".join(_get(xi, span))
                          for id_, xi, span in zip(data_set.data['ids'], data_set.data['x'], spans)}
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = ForwardEvaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), yp2.tolist(), float(loss), id2answer_dict, tensor_dict=tensor_dict)
        return e

    @staticmethod
    def compare(yi, ypi, yp2i):
        for start, stop in yi:
            aypi = argmax(ypi)
            mask = np.zeros(yp2i.shape)
            mask[aypi[0], aypi[1]:] = np.ones([yp2i.shape[1] - aypi[1]])
            if tuple(start) == aypi and (stop[0], stop[1]-1) == argmax(yp2i * mask):
                return True
        return False

    @staticmethod
    def compare2(yi, span):
        for start, stop in yi:
            if tuple(start) == span[0] and tuple(stop) == span[1]:
                return True
        return False

    @staticmethod
    def span_f1(yi, span):
        max_f1 = 0
        for start, stop in yi:
            if start[0] == span[0][0]:
                true_span = start[1], stop[1]
                pred_span = span[0][1], span[1][1]
                f1 = span_f1(true_span, pred_span)
                max_f1 = max(f1, max_f1)
        return max_f1


def get_best_span(ypi, yp2i):

    max_val = 0
    best_word_span = (0, 1)
    best_sent_idx = 0
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        argmax_j1 = 0
        for j in range(len(ypif)):
            val1 = ypif[argmax_j1]
            if val1 < ypif[j]:
                val1 = ypif[j]
                argmax_j1 = j

            val2 = yp2if[j]
            if val1 * val2 > max_val:
                best_word_span = (argmax_j1, j)
                best_sent_idx = f
                max_val = val1 * val2
    return (best_sent_idx, best_word_span[0]), (best_sent_idx, best_word_span[1] + 1)


def get_span_score_pairs(ypi, yp2i):
    span_score_pairs = []
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        for j in range(len(ypif)):
            for k in range(j, len(yp2if)):
                span = ((f, j), (f, k+1))
                score = ypif[j] * yp2if[k]
                span_score_pairs.append((span, score))
    return span_score_pairs
