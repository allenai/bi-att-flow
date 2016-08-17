import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm

from basic.evaluator import TempEvaluator
from basic.graph_handler import GraphHandler
from basic.model import Model
from basic.trainer import Trainer

from basic.read_data import load_metadata, read_data, get_squad_data_filter


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
        config.num_steps = 2
        config.eval_period = 1
        config.log_period = 1
        config.save_period = 1
        config.eval_num_batches = 1


def _train(config):
    load_metadata(config, 'train')  # this updates the config file according to metadata file
    _config_draft(config)
    pprint(config.__flags, indent=2)

    data_filter = get_squad_data_filter(config)
    train_data = read_data(config, 'train', data_filter=data_filter)
    dev_data = read_data(config, 'dev', data_filter=data_filter)

    # construct model graph and variables (using default graph)
    model = Model(config)
    trainer = Trainer(config, model)
    evaluator = TempEvaluator(config, model)
    graph_handler = GraphHandler(config)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    sess = tf.Session()
    graph_handler.initialize(sess)

    # begin training
    num_steps = config.num_steps or int(config.num_epochs * train_data.num_examples / config.batch_size)
    for _, batch in tqdm(train_data.get_batches(config.batch_size, num_batches=num_steps, shuffle=True), total=num_steps):
        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        get_summary = global_step % config.log_period == 0
        loss, summary, train_op = trainer.step(sess, batch, get_summary=get_summary)
        if get_summary:
            graph_handler.add_summary(summary, global_step)

        # Occasional evaluation and saving
        if global_step % config.eval_period == 0:
            num_batches = config.eval_num_batches or math.ceil(dev_data.num_examples / config.batch_size)
            e = evaluator.get_evaluation_from_batches(
                sess, tqdm(dev_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
            graph_handler.add_summaries(e.summaries, global_step)
            graph_handler.dump_eval(e)
            print(e)
        if global_step % config.save_period == 0 or global_step == num_steps:
            graph_handler.save(sess, global_step=global_step)


def _test(config):
    load_metadata(config, 'test')  # this updates the config file according to metadata file
    _config_draft(config)
    # pprint(config.__flag, indent=2)

    test_data = read_data(config, 'test')

    model = Model(config)
    evaluator = TempEvaluator(config, model)
    graph_handler = GraphHandler(config)  # controls all tensors and variables in the graph, including loading /saving

    sess = tf.Session()
    graph_handler.initialize(sess)

    num_batches = config.eval_num_batches or math.ceil(test_data.num_examples / config.batch_size)
    e = evaluator.get_evaluation_from_batches(sess, tqdm(test_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
    graph_handler.dump_eval(e)
    print(e)


def _forward(config):
    load_metadata(config, 'forward')
    _config_draft(config)
    pprint(config.__flag, indent=2)

    forward_data = read_data(config, 'forward')

    model = Model(config)
    evaluator = TempEvaluator(config, model)
    graph_handler = GraphHandler(config)  # controls all tensors and variables in the graph, including loading /saving

    sess = tf.Session()
    graph_handler.initialize(sess)

    num_batches = 1 if config.draft else math.ceil(forward_data.num_examples / config.batch_size)
    e = evaluator.get_evaluation_from_batches(sess, tqdm(forward_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
    graph_handler.dump_eval(e)
    print(e)


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
