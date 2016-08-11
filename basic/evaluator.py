import json
import os

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


class Evaluator(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def get_evaluation(self, sess, data_set):
        feed_dict = self.model.get_feed_dict(data_set, supervised=False)
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
        feed_dict = self.model.get_feed_dict(data_set)
        global_step, yp, loss = sess.run([self.model.global_step, self.model.yp, self.model.loss], feed_dict=feed_dict)
        y = np.array(data_set.data['Y'])
        yp = yp[:data_set.num_examples]
        correct = np.argmax(yp, 1) == y
        e = AccuracyEvaluation(data_set.data_type, int(global_step), yp.tolist(), correct.tolist(), float(loss))
        return e

