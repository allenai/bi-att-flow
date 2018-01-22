import tensorflow as tf

from basic.model import Model
from my.tensorflow import average_gradients


class Trainer(object):
    def __init__(self, config, model):
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdamOptimizer(config.init_lr)
        self.loss = model.get_loss()
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, get_summary=False):
        assert isinstance(sess, tf.Session)
        _, ds = batch
        feed_dict = self.model.get_feed_dict(ds, True)
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op


class MultiGPUTrainer(object):
    def __init__(self, config, models):
        model = models[0]
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdamOptimizer(config.init_lr)
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.models = models
        losses = []
        grads_list = []
        for gpu_idx, model in enumerate(models):
            with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                loss = model.get_loss()
                grads = self.opt.compute_gradients(loss, var_list=self.var_list)
                losses.append(loss)
                grads_list.append(grads)

        self.loss = tf.add_n(losses)/len(losses)
        self.grads = average_gradients(grads_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def step(self, sess, batches, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for batch, model in zip(batches, self.models):
            _, ds = batch
            feed_dict.update(model.get_feed_dict(ds, True))

        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op
