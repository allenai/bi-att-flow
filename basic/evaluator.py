import json
import os

import itertools
import numpy as np
import tensorflow as tf

from basic.read_data import DataSet


class Evaluation(object):
    def __init__(self, data_type, global_step, yp):
        self.data_type = data_type
        self.global_step = global_step
        self.yp = yp
        self.num_examples = len(yp)
        self.dict = {'data_type': data_type,
                     'global_step': global_step,
                     'yp': yp,
                     'num_examples': self.num_examples}
        self.summary = None

    def __repr__(self):
        return "{} step {}".format(self.data_type, self.global_step)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        return Evaluation(self.data_type, self.global_step, new_yp)

    def __radd__(self, other):
        return self.__add__(other)


class AccuracyEvaluation(Evaluation):
    def __init__(self, data_type, global_step, yp, correct, loss):
        super(AccuracyEvaluation, self).__init__(data_type, global_step, yp)
        self.loss = loss
        self.correct = correct
        self.acc = sum(correct) / len(correct)
        self.dict['loss'] = loss
        self.dict['correct'] = correct
        self.dict['acc'] = self.acc
        value = tf.Summary.Value(tag='dev/loss', simple_value=self.loss)
        self.summary = tf.Summary(value=[value])

    def __repr__(self):
        return "{} step {}: accuracy={}, loss={}".format(self.data_type, self.global_step, self.acc, self.loss)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        new_correct = self.correct + other.correct
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_correct)
        return AccuracyEvaluation(self.data_type, self.global_step, new_yp, new_correct, new_loss)


class F1Evaluation(Evaluation):
    def __init__(self, data_type, global_step, y, yp, loss):
        super(F1Evaluation, self).__init__(data_type, global_step, yp)
        self.y = y
        self.loss = loss
        self.num_tps = [sum(np.array(yi) & np.array(ypi)) for yi, ypi in zip(y, yp)]
        self.precs = [num_tp / (sum(ypi) or 1) for num_tp, ypi in zip(self.num_tps, yp)]
        self.recalls = [num_tp / sum(yi) for num_tp, yi in zip(self.num_tps, y)]
        self.f1s = [0 if p+r == 0 else 2*p*r/(p+r) for p, r in zip(self.precs, self.recalls)]
        self.prec = sum(self.precs) / len(self.precs)
        self.recall = sum(self.recalls) / len(self.recalls)
        self.f1 = sum(self.f1s) / len(self.f1s)
        self.dict['loss'] = loss
        self.dict['precs'] = self.precs
        self.dict['recalls'] = self.recalls
        self.dict['f1s'] = self.f1s
        value = tf.Summary.Value(tag='dev/loss', simple_value=self.loss)
        self.summary = tf.Summary(value=[value])

    def __repr__(self):
        return "{} step {}: prec={:.2f}, recall={:.2f}, f1={:.2f}, loss={:.2f}".format(
            self.data_type, self.global_step, self.prec, self.recall, self.f1, self.loss
        )

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_y = self.y + other.y
        new_yp = self.yp + other.yp
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_y)
        return F1Evaluation(self.data_type, self.global_step, new_y, new_yp, new_loss)


class Evaluator(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def get_evaluation(self, sess, data_set):
        feed_dict = self.model.get_feed_dict(data_set, False, supervised=False)
        global_step, yp = sess.run([self.model.global_step, self.model.yp], feed_dict=feed_dict)
        yp = yp[:data_set.num_examples]
        e = Evaluation(data_set.data_type, int(global_step), yp.tolist())
        return e

    def get_evaluation_from_batches(self, sess, batches):
        e = sum(self.get_evaluation(sess, batch) for batch in batches)
        return e


class AccuracyEvaluator(Evaluator):
    def get_evaluation(self, sess, data_set):
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)
        global_step, yp, loss = sess.run([self.model.global_step, self.model.yp, self.model.loss], feed_dict=feed_dict)
        y = np.array(data_set.data['y'])
        yp = yp[:data_set.num_examples]
        correct = np.argmax(yp, 1) == y
        e = AccuracyEvaluation(data_set.data_type, int(global_step), yp.tolist(), correct.tolist(), float(loss))
        return e


class F1Evaluator(Evaluator):
    def get_evaluation(self, sess, data_set):
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)
        global_step, yp, loss = sess.run([self.model.global_step, self.model.yp, self.model.loss], feed_dict=feed_dict)
        yp = (yp[:data_set.num_examples] >= 0.5).tolist()
        y = [[[j == start_idx[0] and start_idx[1] <= k < stop_idx[1]
               for k in range(len(ypij))] for j, ypij in enumerate(ypi)]
             for (start_idx, stop_idx), ypi in zip(data_set.data['y'], yp)]
        yp = list(list(itertools.chain(*ypi)) for ypi in yp)
        y = list(list(itertools.chain(*yi)) for yi in y)
        e = F1Evaluation(data_set.data_type, int(global_step), y, yp, float(loss))
        return e
