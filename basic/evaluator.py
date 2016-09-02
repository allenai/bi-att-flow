import numpy as np
import tensorflow as tf
import os

from basic.read_data import DataSet
from my.nltk_utils import span_f1
from my.utils import argmax


class Evaluation(object):
    def __init__(self, data_type, global_step, idxs, yp):
        self.data_type = data_type
        self.global_step = global_step
        self.idxs = idxs
        self.yp = yp
        self.num_examples = len(yp)
        self.dict = {'data_type': data_type,
                     'global_step': global_step,
                     'yp': yp,
                     'idxs': idxs,
                     'num_examples': self.num_examples}
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
        return Evaluation(self.data_type, self.global_step, new_idxs, new_yp)

    def __radd__(self, other):
        return self.__add__(other)


class LabeledEvaluation(Evaluation):
    def __init__(self, data_type, global_step, idxs, yp, y):
        super(LabeledEvaluation, self).__init__(data_type, global_step, idxs, yp)
        self.y = y
        self.dict['y'] = y

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        new_y = self.y + other.y
        new_idxs = self.idxs + other.idxs
        return LabeledEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_y)


class AccuracyEvaluation(LabeledEvaluation):
    def __init__(self, data_type, global_step, idxs, yp, y, correct, loss):
        super(AccuracyEvaluation, self).__init__(data_type, global_step, idxs, yp, y)
        self.loss = loss
        self.correct = correct
        self.acc = sum(correct) / len(correct)
        self.dict['loss'] = loss
        self.dict['correct'] = correct
        self.dict['acc'] = self.acc
        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='dev/loss', simple_value=self.loss)])
        acc_summary = tf.Summary(value=[tf.Summary.Value(tag='dev/acc', simple_value=self.acc)])
        self.summaries = [loss_summary, acc_summary]

    def __repr__(self):
        return "{} step {}: accuracy={}, loss={}".format(self.data_type, self.global_step, self.acc, self.loss)

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
        return AccuracyEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_y, new_correct, new_loss)


class Evaluator(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        feed_dict = self.model.get_feed_dict(data_set, False, supervised=False)
        global_step, yp = sess.run([self.model.global_step, self.model.yp], feed_dict=feed_dict)
        yp = yp[:data_set.num_examples]
        e = Evaluation(data_set.data_type, int(global_step), idxs, yp.tolist())
        return e

    def get_evaluation_from_batches(self, sess, batches):
        e = sum(self.get_evaluation(sess, batch) for batch in batches)
        return e


class LabeledEvaluator(Evaluator):
    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        feed_dict = self.model.get_feed_dict(data_set, False, supervised=False)
        global_step, yp = sess.run([self.model.global_step, self.model.yp], feed_dict=feed_dict)
        yp = yp[:data_set.num_examples]
        y = feed_dict[self.model.y]
        e = LabeledEvaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), y.tolist())
        return e


class AccuracyEvaluator(LabeledEvaluator):
    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)
        global_step, yp, loss = sess.run([self.model.global_step, self.model.yp, self.model.loss], feed_dict=feed_dict)
        y = data_set.data['y']
        yp = yp[:data_set.num_examples]
        correct = [self.__class__.compare(yi, ypi) for yi, ypi in zip(y, yp)]
        e = AccuracyEvaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), y, correct, float(loss))
        return e

    @staticmethod
    def compare(yi, ypi):
        for start, stop in yi:
            if start == int(np.argmax(ypi)):
                return True
        return False


class AccuracyEvaluator2(AccuracyEvaluator):
    @staticmethod
    def compare(yi, ypi):
        for start, stop in yi:
            para_start = int(np.argmax(np.max(ypi, 1)))
            sent_start = int(np.argmax(ypi[para_start]))
            if tuple(start) == (para_start, sent_start):
                return True
        return False


class TempEvaluation(AccuracyEvaluation):
    def __init__(self, data_type, global_step, idxs, yp, yp2, y, correct, loss, f1s, id2answer_dict):
        super(TempEvaluation, self).__init__(data_type, global_step, idxs, yp, y, correct, loss)
        self.yp2 = yp2
        self.f1s = f1s
        self.f1 = float(np.mean(f1s))
        self.dict['yp2'] = yp2
        self.dict['f1s'] = f1s
        self.dict['f1'] = self.f1
        self.id2answer_dict = id2answer_dict
        f1_summary = tf.Summary(value=[tf.Summary.Value(tag='dev/f1', simple_value=self.f1)])
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
        return TempEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_yp2, new_y, new_correct, new_loss, new_f1s, new_id2answer_dict)

    def __repr__(self):
        return "{} step {}: accuracy={:.4f}, f1={:.4f}, loss={:.4f}".format(self.data_type, self.global_step, self.acc, self.f1, self.loss)


class TempEvaluator(LabeledEvaluator):
    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)
        global_step, yp, yp2, loss = sess.run([self.model.global_step, self.model.yp, self.model.yp2, self.model.loss], feed_dict=feed_dict)
        y = data_set.data['y']
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
        e = TempEvaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), yp2.tolist(), y,
                           correct, float(loss), f1s, id2answer_dict)
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
    best_word_span = None
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





