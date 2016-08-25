import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from tree.evaluator import AccuracyEvaluator2, Evaluator
from tree.graph_handler import GraphHandler
from tree.model import Model
from tree.trainer import Trainer

from tree.read_data import load_metadata, read_data, get_squad_data_filter, update_config


def main(config):
    set_dirs(config)
    if config.mode == 'train':
        _train(config)
    elif config.mode == 'test':
        _test(config)
    elif config.mode == 'forward':
        _forward(config)
    else:
        raise ValueError("invalid value for 'mode': {}".format(config.mode))


def _config_draft(config):
    if config.draft:
        config.num_steps = 10
        config.eval_period = 10
        config.log_period = 1
        config.save_period = 10
        config.eval_num_batches = 1


def _train(config):
    # load_metadata(config, 'train')  # this updates the config file according to metadata file

    data_filter = get_squad_data_filter(config)
    train_data = read_data(config, 'train', config.load, data_filter=data_filter)
    dev_data = read_data(config, 'dev', True, data_filter=data_filter)
    update_config(config, [train_data, dev_data])

    _config_draft(config)

    word2vec_dict = train_data.shared['lower_word2vec'] if config.lower_word else train_data.shared['word2vec']
    word2idx_dict = train_data.shared['word2idx']
    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
    print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
    config.emb_mat = emb_mat

    # construct model graph and variables (using default graph)
    pprint(config.__flags, indent=2)
    model = Model(config)
    trainer = Trainer(config, model)
    evaluator = AccuracyEvaluator2(config, model)
    graph_handler = GraphHandler(config)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    sess = tf.Session()
    graph_handler.initialize(sess)

    # begin training
    num_steps = config.num_steps or int(config.num_epochs * train_data.num_examples / config.batch_size)
    max_acc = 0
    noupdate_count = 0
    global_step = 0
    for _, batch in tqdm(train_data.get_batches(config.batch_size, num_batches=num_steps, shuffle=True), total=num_steps):
        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        get_summary = global_step % config.log_period == 0
        loss, summary, train_op = trainer.step(sess, batch, get_summary=get_summary)
        if get_summary:
            graph_handler.add_summary(summary, global_step)

        # Occasional evaluation and saving
        if global_step % config.save_period == 0:
            graph_handler.save(sess, global_step=global_step)
        if global_step % config.eval_period == 0:
            num_batches = math.ceil(dev_data.num_examples / config.batch_size)
            if 0 < config.eval_num_batches < num_batches:
                num_batches = config.eval_num_batches
            e = evaluator.get_evaluation_from_batches(
                sess, tqdm(dev_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
            graph_handler.add_summaries(e.summaries, global_step)
            if e.acc > max_acc:
                max_acc = e.acc
                noupdate_count = 0
            else:
                noupdate_count += 1
                if noupdate_count == config.early_stop:
                    break
            if config.dump_eval:
                graph_handler.dump_eval(e)
    if global_step % config.save_period != 0:
        graph_handler.save(sess, global_step=global_step)


def _test(config):
    test_data = read_data(config, 'test', True)
    update_config(config, [test_data])

    _config_draft(config)

    pprint(config.__flags, indent=2)
    model = Model(config)
    evaluator = AccuracyEvaluator2(config, model)
    graph_handler = GraphHandler(config)  # controls all tensors and variables in the graph, including loading /saving

    sess = tf.Session()
    graph_handler.initialize(sess)

    num_batches = math.ceil(test_data.num_examples / config.batch_size)
    if 0 < config.eval_num_batches < num_batches:
        num_batches = config.eval_num_batches
    e = evaluator.get_evaluation_from_batches(sess, tqdm(test_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
    print(e)
    if config.dump_eval:
        graph_handler.dump_eval(e)


def _forward(config):

    forward_data = read_data(config, 'forward', True)

    _config_draft(config)

    pprint(config.__flag, indent=2)
    model = Model(config)
    evaluator = Evaluator(config, model)
    graph_handler = GraphHandler(config)  # controls all tensors and variables in the graph, including loading /saving

    sess = tf.Session()
    graph_handler.initialize(sess)

    num_batches = math.ceil(forward_data.num_examples / config.batch_size)
    if 0 < config.eval_num_batches < num_batches:
        num_batches = config.eval_num_batches
    e = evaluator.get_evaluation_from_batches(sess, tqdm(forward_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
    print(e)
    if config.dump_eval:
        graph_handler.dump_eval(e)


def set_dirs(config):
    # create directories
    if not config.load and os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)

    config.save_dir = os.path.join(config.out_dir, "save")
    config.log_dir = os.path.join(config.out_dir, "log")
    config.eval_dir = os.path.join(config.out_dir, "eval")
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.eval_dir)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser.parse_args()


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _run():
    args = _get_args()
    with open(args.config_path, 'r') as fh:
        config = Config(**json.load(fh))
        main(config)


if __name__ == "__main__":
    _run()
