import numpy as np
import tensorflow as tf
import os

from basic.read_data import DataSet
from my.nltk_utils import span_f1


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
    def __init__(self, data_type, global_step, idxs, yp, yp2, y, correct, loss, f1s):
        super(TempEvaluation, self).__init__(data_type, global_step, idxs, yp, y, correct, loss)
        self.yp2 = yp2
        self.f1s = f1s
        self.f1 = float(np.mean(f1s))
        self.dict['yp2'] = yp2
        self.dict['f1s'] = f1s
        self.dict['f1'] = self.f1
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
        return TempEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_yp2, new_y, new_correct, new_loss, new_f1s)


class TempEvaluator(LabeledEvaluator):
    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)
        global_step, yp, yp2, loss = sess.run([self.model.global_step, self.model.yp, self.model.yp2, self.model.loss], feed_dict=feed_dict)
        y = data_set.data['y']
        yp, yp2 = yp[:data_set.num_examples], yp2[:data_set.num_examples]
        correct = [self.__class__.compare(yi, ypi, yp2i) for yi, ypi, yp2i in zip(y, yp, yp2)]
        f1s = [self.__class__.span_f1(yi, ypi, yp2i) for yi, ypi, yp2i in zip(y, yp, yp2)]
        e = TempEvaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), yp2.tolist(), y,
                           correct, float(loss), f1s)
        return e

    @staticmethod
    def compare(yi, ypi, yp2i):
        for start, stop in yi:
            if tuple(start) == argmax(ypi) and tuple(stop) == argmax(yp2i):
                return True
        return False

    @staticmethod
    def span_f1(yi, ypi, yp2i):
        aypi = argmax(ypi)
        ayp2i = argmax(yp2i)
        max_f1 = 0
        for start, stop in yi:
            if start == aypi[0] == ayp2i[0]:
                true_span = (start, stop)
                pred_span = aypi[1], ayp2i[1] + 1
                f1 = span_f1(true_span, pred_span)
                max_f1 = max(f1, max_f1)
        return max_f1


def argmax(x):
    i0 = int(np.argmax(np.max(x, 1)))
    i1 = int(np.argmax(x[i0]))
    return i0, i1

