import tensorflow as tf

from basic.model import Model
from my.tensorflow import average_gradients


class Trainer(object):
    def __init__(self, config, model):
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
        self.loss = model.get_loss()
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.ema_op = model.ema_op
        self.summary = model.summary
        self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        opt_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

        # Define train op
        with tf.control_dependencies([opt_op]):
            self.train_op = tf.group(self.ema_op)

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self._get_feed_dict(batch)
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op

    def _get_feed_dict(self, batch):
        _, data_set = batch
        return self.model.get_feed_dict(data_set, True)


class MultiGPUTrainer(Trainer):
    def __init__(self, config, models):
        self.models = models
        super(MultiGPUTrainer, self).__init__(config, models[0])
        losses = [model.get_loss() for model in models]
        grads_list = []
        for gpu_idx, loss in enumerate(losses):
            with tf.name_scope("gpu_{}".format(gpu_idx)), tf.device("/gpu:{}".format(gpu_idx)):
                grads = self.opt.compute_gradients(loss, var_list=self.var_list)
            grads_list.append(grads)

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
        for model, (_, data_set) in zip(self.models, batches):
            feed_dict.update(model.get_feed_dict(data_set, True))
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

