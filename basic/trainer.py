import tensorflow as tf
from tensorflow.models.image.cifar10.cifar10_multi_gpu_train import average_gradients

from basic.model import Model


class Trainer(object):
    @staticmethod
    def get_multi_gpu_trainer(config, models):
        model = models[0]
        trainer = Trainer(config, model)
        losses = [model.get_loss() for model in models]
        grads_list = [trainer.opt.compute_gradients(loss, var_list=trainer.var_list) for loss in losses]
        with tf.name_scope("average"), tf.device("/cpu:0"):
            trainer.loss = tf.add_n(losses)/len(losses)
            trainer.grads = average_gradients(grads_list)
        opt_op = trainer.opt.apply_gradients(trainer.grads, global_step=self.global_step)

        # Define train op
        with tf.control_dependencies([opt_op]):
            trainer.train_op = tf.group(trainer.ema_op)

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
        feed_dict = self.model.get_feed_dict(batch, True)
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op
