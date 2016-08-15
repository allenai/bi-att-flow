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


class LabeledEvaluation(Evaluation):
    def __init__(self, data_type, global_step, yp, y):
        super(LabeledEvaluation, self).__init__(data_type, global_step, yp)
        self.y = y

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        new_y = self.y + other.y
        return LabeledEvaluation(self.data_type, self.global_step, new_yp, new_y)


class AccuracyEvaluation(LabeledEvaluation):
    def __init__(self, data_type, global_step, yp, y, correct, loss):
        super(AccuracyEvaluation, self).__init__(data_type, global_step, yp, y)
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
        new_y = self.y + other.y
        new_correct = self.correct + other.correct
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_correct)
        return AccuracyEvaluation(self.data_type, self.global_step, new_yp, new_y, new_correct, new_loss)


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


class LabeledEvaluator(Evaluator):
    def get_evaluation(self, sess, data_set):
        feed_dict = self.model.get_feed_dict(data_set, False, supervised=False)
        global_step, yp = sess.run([self.model.global_step, self.model.yp], feed_dict=feed_dict)
        yp = yp[:data_set.num_examples]
        y = feed_dict[self.model.y]
        e = LabeledEvaluation(data_set.data_type, int(global_step), yp.tolist(), y.tolist())
        return e


class AccuracyEvaluator(LabeledEvaluator):
    def get_evaluation(self, sess, data_set):
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)
        global_step, yp, loss = sess.run([self.model.global_step, self.model.yp, self.model.loss], feed_dict=feed_dict)
        y = feed_dict[self.model.y]
        yp = yp[:data_set.num_examples]
        correct = [self.__class__.compare(yi, ypi) for yi, ypi in zip(y, yp)]
        e = AccuracyEvaluation(data_set.data_type, int(global_step), yp.tolist(), y.tolist(), correct, float(loss))
        return e

    @staticmethod
    def compare(yi, ypi):
        return int(np.argmax(yi)) == int(np.argmax(ypi))


class AccuracyEvaluator2(AccuracyEvaluator):
    @staticmethod
    def compare(yi, ypi):
        i = int(np.argmax(yi.flatten()))
        j = int(np.argmax(ypi.flatten()))
        # print(i, j, i == j)
        return i == j
