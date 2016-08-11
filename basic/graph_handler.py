import json
import os

import tensorflow as tf

from basic.evaluator import Evaluation


class GraphHandler(object):
    def __init__(self, config):
        self.config = config
        self.saver = tf.train.Saver()
        self.writer = None
        self.save_path = os.path.join(config.save_dir, config.model_name)

    def initialize(self, sess):
        if self.config.load:
            self._load(sess)
        else:
            sess.run(tf.initialize_all_variables())

        if self.config.mode == 'train':
            self.writer = tf.train.SummaryWriter(self.config.log_dir)

    def save(self, sess, global_step=None):
        self.saver.save(sess, self.save_path, global_step=global_step)

    def _load(self, sess):
        save_dir = self.config.save_dir
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        assert checkpoint is not None, "cannot load checkpoint at {}".format(save_dir)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

    def add_summary(self, summary, global_step):
        self.writer.add_summary(summary, global_step)

    def dump_eval(self, e):
        assert isinstance(e, Evaluation)
        path = os.path.join(self.config.eval_dir, "{}-{}.json".format(e.data_type, str(e.global_step).zfill(3)))
        with open(path, 'w') as fh:
            json.dump(e.dict, fh)

