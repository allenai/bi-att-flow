import itertools
import json
import logging
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf

from model.read_data import DataSet, NUM
from my.tensorflow import average_gradients
from my.utils import mytqdm


class BaseRunner(object):
    def __init__(self, params, sess, towers):
        assert isinstance(sess, tf.Session)
        self.sess = sess
        self.params = params
        self.towers = towers
        self.ref_tower = towers[0]
        self.num_towers = len(towers)
        self.placeholders = {}
        self.tensors = {}
        self.saver = None
        self.writer = None
        self.initialized = False
        self.train_ops = {}
        self.write_log = params.write_log

    def initialize(self):
        params = self.params
        sess = self.sess
        device_type = params.device_type
        summaries = []

        global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                      initializer=tf.constant_initializer(0), trainable=False)
        self.tensors['global_step'] = global_step

        epoch = tf.get_variable('epoch', shape=[], dtype='int32',
                                initializer=tf.constant_initializer(0), trainable=False)
        self.tensors['epoch'] = epoch
        with tf.variable_scope("towers"):
            for device_id, tower in enumerate(self.towers):
                with tf.device("/%s:%d" % (device_type, device_id)), tf.name_scope("%s_%d" % (device_type, device_id)):
                    tower.initialize()
                    tf.get_variable_scope().reuse_variables()
        self.initialized = True

        learning_rate = tf.placeholder('float32', name='learning_rate')
        summaries.append(tf.scalar_summary("learning_rate", learning_rate))
        self.placeholders['learning_rate'] = learning_rate

        saver = tf.train.Saver(tf.all_variables())
        self.saver = saver

        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        if not params.supervise:
            summary_op = tf.merge_summary(summaries)
            self.tensors['summary'] = summary_op
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            if self.write_log:
                self.writer = tf.train.SummaryWriter(params.log_dir, sess.graph)

            return

        if params.opt == 'basic':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif params.opt == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif params.opt == 'adam':
            opt = tf.train.AdamOptimizer()
        else:
            raise Exception()

        grads_pairs_dict = defaultdict(list)
        correct_tensors = []
        wrong_tensors = []
        loss_tensors = []
        for tower in self.towers:
            with tf.device("/%s:%d" % (device_type, device_id)), tf.name_scope("%s_%d" % (device_type, device_id)):
                loss_tensor = tower.get_loss_tensor()
                loss_tensors.append(loss_tensor)
                correct_tensor = tower.get_correct_tensor()
                correct_tensors.append(correct_tensor)
                wrong_tensor = tower.get_wrong_tensor()
                wrong_tensors.append(wrong_tensor)

                for key, variables in tower.variables_dict.items():
                    grads_pair = opt.compute_gradients(loss_tensor, var_list=variables)
                    grads_pairs_dict[key].append(grads_pair)

        with tf.name_scope("gpu_sync"):
            loss_tensor = tf.reduce_mean(tf.pack(loss_tensors), 0, name='loss')
            correct_tensor = tf.concat(0, correct_tensors, name="correct")
            wrong_tensor = tf.concat(0, wrong_tensors, name="wrong")
            with tf.name_scope("average_gradients"):
                grads_pair_dict = {key: average_gradients(grads_pairs)
                                   for key, grads_pairs in grads_pairs_dict.items()}
                if params.max_grad_norm:
                    grads_pair_dict = {key: [(tf.clip_by_norm(grad, params.max_grad_norm), var)
                                             for grad, var in grads_pair]
                                       for key, grads_pair in grads_pair_dict.items()}

        self.tensors['loss'] = loss_tensor
        self.tensors['correct'] = correct_tensor
        self.tensors['wrong'] = wrong_tensor
        summaries.append(tf.scalar_summary(loss_tensor.op.name, loss_tensor))

        for key, grads_pair in grads_pair_dict.items():
            for grad, var in grads_pair:
                if grad is not None:
                    summaries.append(tf.histogram_summary(var.op.name+'/gradients/'+key, grad))

        apply_grads_op_dict = {key: opt.apply_gradients(grads_pair, global_step=global_step)
                               for key, grads_pair in grads_pair_dict.items()}

        self.train_ops = {key: tf.group(apply_grads_op)
                          for key, apply_grads_op in apply_grads_op_dict.items()}

        summary_op = tf.merge_summary(summaries)
        self.tensors['summary'] = summary_op
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        if self.write_log:
            self.writer = tf.train.SummaryWriter(params.log_dir, sess.graph)

    def _get_feed_dict(self, batches, mode, **kwargs):
        placeholders = self.placeholders
        learning_rate_ph = placeholders['learning_rate']
        learning_rate = kwargs['learning_rate'] if mode == 'train' else 0.0
        feed_dict = {learning_rate_ph: learning_rate}
        for tower_idx, tower in enumerate(self.towers):
            batch = batches[tower_idx] if tower_idx < len(batches) else None
            cur_feed_dict = tower.get_feed_dict(batch, mode, **kwargs)
            feed_dict.update(cur_feed_dict)
        return feed_dict

    def _train_batches(self, batches, **kwargs):
        sess = self.sess
        tensors = self.tensors
        feed_dict = self._get_feed_dict(batches, 'train', **kwargs)
        train_op = self._get_train_op(**kwargs)
        ops = [train_op, tensors['summary'], tensors['global_step']]
        train, summary, global_step = sess.run(ops, feed_dict=feed_dict)
        return train, summary, global_step

    def _eval_batches(self, batches, eval_tensor_names=(), **eval_kwargs):
        sess = self.sess
        tensors = self.tensors
        params = self.params
        feed_dict = self._get_feed_dict(batches, 'eval', **eval_kwargs)
        ops = [tensors[name] for name in ['summary', 'global_step']]
        summary, global_step = sess.run(ops, feed_dict=feed_dict)
        if params.supervise:
            ops = [tensors[name] for name in ['correct', 'wrong', 'loss']]
            correct, wrong, loss = sess.run(ops, feed_dict=feed_dict)
            num_corrects = np.sum(correct)
            num_wrongs = np.sum(wrong)
        else:
            num_corrects, num_wrongs, loss = None, None, None

        if len(eval_tensor_names) > 0:
            valuess = [sess.run([tower.tensors[name] for name in eval_tensor_names], feed_dict=feed_dict)
                       for tower in self.towers]
            valuess = [[each_values[:batch[NUM]] for name, each_values in zip(eval_tensor_names, values)]
                       for values, batch in zip(valuess, batches)]
        else:
            valuess = [[]]

        return (summary, global_step), (num_corrects, num_wrongs, loss), valuess

    def train(self, train_data_set, num_epochs, val_data_set=None, eval_ph_names=(),
              eval_tensor_names=(), num_batches=None, val_num_batches=None):
        assert isinstance(train_data_set, DataSet)
        assert self.initialized, "Initialize tower before training."

        sess = self.sess
        writer = self.writer
        params = self.params
        progress = params.progress
        val_acc = None
        # if num batches is specified, then train only that many
        num_batches = num_batches or train_data_set.get_num_batches(partial=False)
        num_iters_per_epoch = int(num_batches / self.num_towers)
        num_digits = int(np.log10(num_batches))

        epoch_op = self.tensors['epoch']
        epoch = sess.run(epoch_op)
        string = "starting from epoch {}/{}, {} batches per epoch.".format(epoch+1, num_epochs, num_iters_per_epoch)
        logging.info(string)
        print(string)
        while epoch < num_epochs:
            train_args = self._get_train_args(epoch)
            string = "epoch {}|".format(str(epoch+1).zfill(num_digits))
            train_data_set.reset()
            for _ in mytqdm(range(num_iters_per_epoch), desc=string, show=progress):
                batches = [train_data_set.get_next_batch() for _ in range(self.num_towers)]
                _, summary, global_step = self._train_batches(batches, **train_args)
                if self.write_log:
                    writer.add_summary(summary, global_step)

            train_data_set.num_epochs_completed += 1
            assign_op = epoch_op.assign_add(1)
            _, epoch = sess.run([assign_op, epoch_op])

            if val_data_set and epoch % params.val_period == 0:
                self.eval(train_data_set, eval_tensor_names=eval_tensor_names, num_batches=val_num_batches)
                val_loss, val_acc = self.eval(val_data_set, eval_tensor_names=eval_tensor_names, num_batches=val_num_batches)

            if epoch % params.save_period == 0:
                self.save()

        return val_loss, val_acc

    def eval(self, data_set, eval_tensor_names=(), eval_ph_names=(), num_batches=None):
        # TODO : eval_ph_names
        assert isinstance(data_set, DataSet)
        assert self.initialized, "Initialize tower before training."

        params = self.params
        sess = self.sess
        epoch_op = self.tensors['epoch']
        epoch = sess.run(epoch_op)
        progress = params.progress
        num_batches = num_batches or data_set.get_num_batches(partial=True)
        num_iters = int(np.ceil(num_batches / self.num_towers))
        num_corrects, num_wrongs, total_loss = 0, 0, 0.0
        eval_values = []
        idxs = []
        N = data_set.batch_size * num_batches
        if N > data_set.num_examples:
            N = data_set.num_examples
        eval_args = self._get_eval_args(epoch)
        data_set.reset(shuffle=False)
        string = "eval on {}, N={}|".format(data_set.name, N)
        for _ in mytqdm(range(num_iters), desc=string, show=progress):
            batches = []
            for _ in range(self.num_towers):
                if data_set.has_next_batch(partial=True):
                    idxs.extend(data_set.get_batch_idxs(partial=True))
                    batches.append(data_set.get_next_batch(partial=True))
            (_, global_step), (cur_num_corrects, cur_num_wrongs, cur_avg_loss), eval_value_batches = \
                self._eval_batches(batches, eval_tensor_names=eval_tensor_names, **eval_args)
            if params.supervise:
                num_corrects += cur_num_corrects
                num_wrongs += cur_num_wrongs
                cur_num = sum(batch[NUM] for batch in batches)
                total_loss += cur_avg_loss * cur_num
            for eval_value_batch in eval_value_batches:
                eval_values.append([x.tolist() for x in eval_value_batch])  # numpy.array.toList

        # For printing results to stdout and log
        if params.supervise:
            total = num_corrects + num_wrongs
            loss = float(total_loss) / total

            acc = float(num_corrects) / total
            string = "%s at epoch %d: acc = %.2f%% = %d / %d, loss = %.4f" % \
                     (data_set.name, epoch, 100 * acc, num_corrects, total, loss)
            logging.info(string)
            print(string)
        else:
            loss, acc = None, None

        # For outputting eval json files
        if len(eval_tensor_names) > 0:
            ids = [data_set.idx2id[idx] for idx in idxs]
            zipped_eval_values = [list(itertools.chain(*each)) for each in zip(*eval_values)]
            values_dict = {name: values for name, values in zip(eval_tensor_names, zipped_eval_values)}
            out = {'ids': ids, 'values': values_dict}
            eval_path = os.path.join(params.eval_dir, "%s_%s.json" % (data_set.name, str(epoch).zfill(4)))
            json.dump(out, open(eval_path, 'w'))

        return loss, acc

    def _get_train_op(self, **kwargs):
        return self.train_ops['all']

    def _get_train_args(self, epoch_idx):
        params = self.params
        learning_rate = params.init_lr

        anneal_period = params.lr_anneal_period
        anneal_ratio = params.lr_anneal_ratio
        num_periods = int(epoch_idx / anneal_period)
        factor = anneal_ratio ** num_periods
        learning_rate *= factor

        train_args = self._get_common_args(epoch_idx)
        train_args['learning_rate'] = learning_rate
        return train_args

    def _get_eval_args(self, epoch_idx):
        return self._get_common_args(epoch_idx)

    def _get_common_args(self, epoch_idx):
        return {}

    def save(self):
        assert self.initialized, "Initialize tower before saving."

        sess = self.sess
        params = self.params
        save_dir = params.save_dir
        name = params.model_name
        global_step = self.tensors['global_step']
        logging.info("saving model ...")
        save_path = os.path.join(save_dir, name)
        self.saver.save(sess, save_path, global_step)
        logging.info("saving done.")

    def load(self):
        assert self.initialized, "Initialize tower before loading."

        sess = self.sess
        params = self.params
        save_dir = params.save_dir
        logging.info("loading model ...")
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        assert checkpoint is not None, "Cannot load checkpoint at %s" % save_dir
        self.saver.restore(sess, checkpoint.model_checkpoint_path)
        logging.info("loading done.")


class BaseTower(object):
    def __init__(self, params):
        self.params = params
        self.placeholders = {}
        self.tensors = {}
        self.variables_dict = {}
        # this initializer is used for weight init that shouldn't be dependent on input size.
        # for MLP weights, the tensorflow default initializer should be used,
        # i.e. uniform unit scaling initializer.
        self.initializer = tf.truncated_normal_initializer(params.init_mean, params.init_std)

    def initialize(self):
        self._initialize_forward()
        if self.params.supervise:
            self._initialize_supervision()
        self.variables_dict['all'] = tf.trainable_variables()

    def _initialize_forward(self):
        # Actual building for network's forward pass
        # Separated so that GPU assignment can be done here.
        # TODO : self.tensors['loss'] and self.tensors['correct'] must be defined.
        raise NotImplementedError()

    def _initialize_supervision(self):
        # Supervision portion of the network
        # Required for training
        raise NotImplementedError()

    def get_correct_tensor(self):
        return self.tensors['correct']

    def get_wrong_tensor(self):
        return self.tensors['wrong']

    def get_loss_tensor(self):
        return self.tensors['loss']

    def get_variables_dict(self):
        return self.variables_dict

    def get_feed_dict(self, batch, mode, **kwargs):
        return self._get_feed_dict(batch, mode, **kwargs)

    def _get_feed_dict(self, batch, mode, **kwargs):
        # TODO : MUST handle batch = None
        raise NotImplementedError()
