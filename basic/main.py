import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from basic.evaluator import F1Evaluator, ForwardEvaluator, MultiGPUF1Evaluator
from basic.graph_handler import GraphHandler
from basic.model import Model, get_multi_gpu_models
from basic.trainer import Trainer, MultiGPUTrainer

from basic.read_data import load_metadata, read_data, get_squad_data_filter, update_config


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
    # load_metadata(config, 'train')  # this updates the config file according to metadata file

    data_filter = get_squad_data_filter(config)
    train_data = read_data(config, 'train', config.load, data_filter=data_filter)
    dev_data = read_data(config, 'dev', True, data_filter=data_filter)
    # test_data = read_data(config, 'test', True, data_filter=data_filter)
    update_config(config, [train_data, dev_data])

    _config_draft(config)

    # construct model graph and variables (using default graph)
    pprint(config.__flags, indent=2)
    if config.num_gpus == 1:
        model = Model(config)
        trainer = Trainer(config, model)
        evaluator = F1Evaluator(config, model)
    else:
        models = get_multi_gpu_models(config)
        model = models[0]
        trainer = MultiGPUTrainer(config, models)
        evaluator = MultiGPUF1Evaluator(config, models)
    graph_handler = GraphHandler(config)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_handler.initialize(sess)

    # begin training
    num_steps = config.num_steps or int(config.num_epochs * train_data.num_examples / config.batch_size)
    max_acc = 0
    noupdate_count = 0
    global_step = 0
    for batch in tqdm(train_data.get_batches(config.batch_size, num_batches=num_steps, shuffle=True), total=num_steps):
        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        get_summary = global_step % config.log_period == 0
        loss, summary, train_op = trainer.step(sess, batch, get_summary=get_summary)
        if get_summary:
            graph_handler.add_summary(summary, global_step)

        # occasional saving
        if global_step % config.save_period == 0:
            graph_handler.save(sess, global_step=global_step)

        if not config.eval:
            continue
        # Occasional evaluation
        if global_step % config.eval_period == 0:
            num_batches = math.ceil(dev_data.num_examples / config.batch_size)
            if 0 < config.eval_num_batches < num_batches:
                num_batches = config.eval_num_batches
            e_train = evaluator.get_evaluation_from_batches(
                sess, tqdm(train_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches)
            )
            graph_handler.add_summaries(e_train.summaries, global_step)
            e_dev = evaluator.get_evaluation_from_batches(
                sess, tqdm(dev_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
            graph_handler.add_summaries(e_dev.summaries, global_step)
            if e_dev.acc > max_acc:
                max_acc = e_dev.acc
                noupdate_count = 0
            else:
                noupdate_count += 1
                if noupdate_count == config.early_stop:
                    break
            if config.dump_eval:
                graph_handler.dump_eval(e_dev)
            if config.dump_answer:
                graph_handler.dump_answer(e_dev)
    if global_step % config.save_period != 0:
        graph_handler.save(sess, global_step=global_step)


def _test(config):
    assert config.load
    test_data = read_data(config, 'test', True)
    update_config(config, [test_data])

    _config_draft(config)

    pprint(config.__flags, indent=2)
    model = Model(config)
    evaluator = F1Evaluator(config, model)
    graph_handler = GraphHandler(config)  # controls all tensors and variables in the graph, including loading /saving

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    graph_handler.initialize(sess)

    num_batches = math.ceil(test_data.num_examples / config.batch_size)
    if 0 < config.eval_num_batches < num_batches:
        num_batches = config.eval_num_batches
    e = evaluator.get_evaluation_from_batches(sess, tqdm(test_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
    print(e)
    if config.dump_answer:
        print("dumping answer ...")
        graph_handler.dump_answer(e)
    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e)


def _forward(config):
    assert config.load
    test_data = read_data(config, config.forward_name, True)
    update_config(config, [test_data])

    _config_draft(config)

    pprint(config.__flags, indent=2)
    model = Model(config)
    evaluator = ForwardEvaluator(config, model)
    graph_handler = GraphHandler(config)  # controls all tensors and variables in the graph, including loading /saving

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_handler.initialize(sess)

    num_batches = math.ceil(test_data.num_examples / config.batch_size)
    if 0 < config.eval_num_batches < num_batches:
        num_batches = config.eval_num_batches
    e = evaluator.get_evaluation_from_batches(sess, tqdm(test_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
    print(e)
    if config.dump_answer:
        print("dumping answer ...")
        graph_handler.dump_answer(e, path=config.answer_path)
    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e)


def set_dirs(config):
    # create directories
    if not config.load and os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)

    config.save_dir = os.path.join(config.out_dir, "save")
    config.log_dir = os.path.join(config.out_dir, "log")
    config.eval_dir = os.path.join(config.out_dir, "eval")
    config.answer_dir = os.path.join(config.out_dir, "answer")
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(config.answer_dir):
        os.mkdir(config.answer_dir)
    if not os.path.exists(config.eval_dir):
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
