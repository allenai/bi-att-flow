import tensorflow as tf
from tensorflow.models.image.cifar10.cifar10_multi_gpu_train import average_gradients

from basic.evaluator import Evaluator, F1Evaluator
from basic.read_data import DataSet
from basic.trainer import Trainer
from my.utils import grouper


def get_multi_gpu_models(config, class_):
    models = []
    with tf.variable_scope("models", caching_device="/cpu:0"):
        for gpu_idx in config.num_gpus:
            with tf.device("/gpu:{}".format(gpu_idx)), tf.name_scope("gpu_{}".format(gpu_idx)):
                model = class_(config)
                models.append(model)
                tf.get_variable_scope().reuse_variables()
    return models


class MultiGPUTrainer(Trainer):
    def __init__(self, config, models):
        self.models = models
        super(MultiGPUTrainer, self).__init__(config, models[0])
        losses = [model.get_loss() for model in models]
        grads_list = [self.opt.compute_gradients(loss, var_list=self.var_list) for loss in losses]
        with tf.name_scope("average"), tf.device("/cpu:0"):
            self.loss = tf.add_n(losses)/len(losses)
            self.grads = average_gradients(grads_list)
        opt_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

        # Define train op
        with tf.control_dependencies([opt_op]):
            self.train_op = tf.group(self.ema_op)

    def step(self, sess, batches, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for model, batch in zip(self.models, batches):
            feed_dict.update(model.get_feed_dict(batch, True))
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op

    def _get_feed_dict(self, batches):
        feed_dict = {}
        for model, (_, data_set) in zip(self.models, batches):
            feed_dict.update(model.get_feed_dict(data_set, True))
        return feed_dict


class MultiGPUF1Evaluator(F1Evaluator):
    def __init__(self, config, models):
        self.models = models
        super(MultiGPUF1Evaluator, self).__init__(config, models[0])
        with tf.name_scope("concat"), tf.device("/cpu:0"):
            self.yp = tf.concat(0, [model.yp for model in self.models])
            self.yp2 = tf.concat(0, [model.yp2 for model in self.models])
            # FIXME : incorrect loss calculation, due to smaller / empty batches
            self.loss = tf.add_n([model.loss for model in self.models])/len(self.models)

    def _split_batch(self, batches):
        idxs_list, data_sets = zip(*batches)
        idxs = sum(idxs_list, [])
        data_set = sum(data_sets, start=data_sets[0].get_empty())
        return idxs, data_set

    def _get_feed_dict(self, batches):
        feed_dict = {}
        for model, (_, data_set) in zip(self.models, batches):
            feed_dict.update(model.get_feed_dict(data_set, True))
        return feed_dict


class MultiGPUDataSet(DataSet):
    def __init__(self, data, data_type, num_gpus, shared=None, valid_idxs=None):
        super(MultiGPUDataSet, self).__init__(data, data_type, shared=shared, valid_idxs=valid_idxs)
        self.num_gpus = num_gpus

    def get_batches(self, batch_size, num_batches=None, shuffle=False):
        flat_batches = super(MultiGPUDataSet, self).get_batches(batch_size, num_batches=num_batches*self.num_gpus, shuffle=shuffle)
        batches = grouper(flat_batches, self.num_gpus, fillvalue=self.get_empty())
        return batches
