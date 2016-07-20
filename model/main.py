import json
import os
import shutil
import logging
import sys
from pprint import pformat

import tensorflow as tf
import numpy as np
import time

from model.base_model import BaseRunner
from model.model import Tower
from config.get_config import get_config_from_file, get_config
from model.read_data import read_data

flags = tf.app.flags

# File directories
flags.DEFINE_string("model_name", "model", "Model name. This will be used for save, log, and eval names. [model]")
flags.DEFINE_string("data_dir", "data/model/squad", "Data directory [data/model/squad]")

# Training parameters
# These affect result performance
flags.DEFINE_integer("batch_size", 256, "Batch size for each tower. [256]")
flags.DEFINE_float("init_mean", 0, "Initial weight mean [0]")
flags.DEFINE_float("init_std", 1.0, "Initial weight std [1.0]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_integer("lr_anneal_period", 100, "Anneal period [100]")
flags.DEFINE_float("lr_anneal_ratio", 0.5, "Anneal ratio [0.5")
flags.DEFINE_integer("num_epochs", 50, "Total number of epochs for training [50]")
flags.DEFINE_string("opt", 'adagrad', 'Optimizer: basic | adagrad | adam [basic]')
flags.DEFINE_float("wd", 0.001, "Weight decay [0.001]")
flags.DEFINE_integer("max_grad_norm", 0, "Max grad norm. 0 for no clipping [0]")
flags.DEFINE_float("max_val_loss", 0.0, "Max val loss [0.0]")

# Training and testing options
# These do not directly affect result performance (they affect duration though)
flags.DEFINE_boolean("train", True, "Train? False if test. [True]")
flags.DEFINE_boolean("supervise", True, "Supervise? Must be True if train=True. [True]")
flags.DEFINE_integer("val_num_batches", 0, "Val num batches. 0 for max possible. [0]")
flags.DEFINE_integer("train_num_batches", 0, "Train num batches. 0 for max possible [0]")
flags.DEFINE_integer("test_num_batches", 0, "Test num batches. 0 for max possible [0]")
flags.DEFINE_boolean("load", True, "Load from saved model? [True]")
flags.DEFINE_boolean("progress", False, "Show progress bar? [False]")
flags.DEFINE_string("device_type", 'gpu', "cpu | gpu [gpu]")
flags.DEFINE_integer("num_devices", 1, "Number of devices to use. Only for multi-GPU. [1]")
flags.DEFINE_integer("val_period", 10, "Validation period (for display purpose only) [10]")
flags.DEFINE_integer("save_period", 10, "Save period [10]")
flags.DEFINE_string("config_id", 'None', "Config name (e.g. local) to load. 'None' to use config here. [None]")
flags.DEFINE_string("config_ext", ".json", "Config file extension: .json | .tsv [.json]")
flags.DEFINE_integer("num_trials", 1, "Number of trials [1]")
flags.DEFINE_string("seq_id", "None", "Sequence id [None]")
flags.DEFINE_string("run_id", "0", "Run id [0]")
flags.DEFINE_boolean("write_log", True, "Write log? [True]")
flags.DEFINE_string("out_dir", "out", "Out dir [out]")
# TODO : don't erase log folder if not write log

# Debugging
flags.DEFINE_boolean("draft", False, "Draft? (quick initialize) [False]")

# App-specific options
# TODO : Any other options
flags.DEFINE_float("keep_prob", 0.5, "Keep probability of LSTM [0.5]")
flags.DEFINE_bool("finetune", True, "Finetune? [True]")

FLAGS = flags.FLAGS


def _init():
    run_id = FLAGS.run_id.zfill(4)
    run_dir = os.path.join(FLAGS.out_dir, FLAGS.model_name, run_id)
    FLAGS.run_dir = run_dir

    # Logging stuff
    stdout_log_path = os.path.join(run_dir, "stdout.log")

    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(open(stdout_log_path, 'w'))
    ch.setLevel(logging.DEBUG)
    root.addHandler(ch)
    """
    if os.path.exists(run_dir) and FLAGS.train and not FLAGS.load:
        shutil.rmtree(run_dir)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    logging.basicConfig(filename=stdout_log_path, filemode='w', level=logging.DEBUG)
    logging.info("\n"*3 + time.ctime() + "\n"*3)


def _makedirs(config, trial_idx):
    run_dir = config.run_dir
    config_id = str(config.config_id).zfill(2)
    trial_idx = str(trial_idx).zfill(2)
    subdir_name = "-".join([config_id, trial_idx])

    base_dir = os.path.join(run_dir, subdir_name)
    eval_dir = os.path.join(base_dir, "eval")
    log_dir = os.path.join(base_dir, "log")
    save_dir = os.path.join(base_dir, "save")
    config.eval_dir = eval_dir
    config.log_dir = log_dir
    config.save_dir = save_dir

    for dir_ in [eval_dir, log_dir, save_dir]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)


def _load_metadata(config):
    data_dir = config.data_dir
    if config.train:
        train_metadata_path = os.path.join(data_dir, "train_metadata.json")
        train_metadata = json.load(open(train_metadata_path, "r"))
        dev_metadata_path = os.path.join(data_dir, "dev_metadata.json")
        dev_metadata = json.load(open(dev_metadata_path, "r"))
        metadata = {key: max(train, dev) for (key, train), (_, dev) in zip(train_metadata.items(), dev_metadata.items())}
    else:
        test_metadata_path = os.path.join(data_dir, "test_metadata.json")
        metadata = json.load(open(test_metadata_path, "r"))
    # TODO: set other parameters, e.g.
    prior_path = os.path.join(data_dir, "prior.json")
    priors = json.load(open(prior_path, 'r'))
    emb_mat = np.array(priors['emb_mat'], dtype='float32')
    config.emb_mat = emb_mat
    config.max_sent_size = metadata['max_sent_size']
    config.max_num_sents = metadata['max_num_sents']
    config.vocab_size = metadata['vocab_size']
    config.max_ques_size = metadata['max_ques_size']
    config.word_vec_size = metadata['word_vec_size']


def _main(config, num_trials):
    _load_metadata(config)

    # Load data
    if config.train:
        train_ds = read_data(config, 'train')
        dev_ds = read_data(config, 'dev')
    else:
        test_ds = read_data(config, 'test')

    # For quick draft initialize (deubgging).
    if config.draft:
        config.train_num_batches = 1
        config.val_num_batches = 1
        config.test_num_batches = 1
        config.num_epochs = 2
        config.val_period = 1
        config.save_period = 1
        # TODO : Add any other parameter that induces a lot of computations

    # Sanity check
    if config.val_period > config.num_epochs:
        config.val_period = config.num_epochs
        logging.warning("val_period is bigger than num_epochs. Adjusting val_period <- num_epochs.")
    if config.save_period > config.num_epochs:
        config.save_period = config.num_epochs
        logging.warning("save_period is bigger than num_epochs. Adjusting save_period <- num_epochs.")

    logging.info(pformat(config.__dict__, indent=2))

    # TODO : specify eval tensor names to save in evals folder
    eval_tensor_names = ['yp_flat']
    eval_ph_names = []

    def get_best_trial_idx(_val_losses):
        return min(enumerate(_val_losses), key=lambda x: x[1])[0]

    losses, accs = [], []
    for trial_idx in range(1, num_trials+1):
        string = "{}\nTrial {}".format("-"*80, trial_idx)
        logging.info(string)
        print(string)
        _makedirs(config, trial_idx)
        graph = tf.Graph()
        # TODO : initialize BaseTower-subclassed objects
        towers = [Tower(config) for _ in range(config.num_devices)]
        sess = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True))
        # TODO : initialize BaseRunner-subclassed object
        runner = BaseRunner(config, sess, towers)
        default_device = '/gpu:0' if config.num_devices == 1 and config.device_type == 'gpu' else '/cpu:0'
        with graph.as_default(), tf.device(default_device):
            runner.initialize()
            if config.train:
                if config.load:
                    runner.load()
                val_loss, val_acc = runner.train(train_ds, config.num_epochs, val_data_set=dev_ds,
                                                 num_batches=config.train_num_batches,
                                                 val_num_batches=config.val_num_batches, eval_ph_names=eval_ph_names)
                accs.append(val_acc)
                losses.append(val_loss)
            else:
                runner.load()
                test_loss, test_acc = runner.eval(test_ds, eval_tensor_names=eval_tensor_names,
                                                  num_batches=config.test_num_batches, eval_ph_names=eval_ph_names)
                losses.append(test_loss)
                accs.append(test_acc)

        if config.supervise:
            best_trial_idx = get_best_trial_idx(losses)
            string = "{}\nMin loss: {:.4f} at Trial {}/{}".format("-"*80, min(losses), best_trial_idx+1, num_trials)
            logging.info(string)
            print(string)

    summary = ""
    return summary


def main(_):
    _init()
    this_dir = os.path.dirname(os.path.realpath(__file__))
    if FLAGS.seq_id == 'None':
        seq = [[FLAGS.config_id, FLAGS.num_trials]]
    else:
        seqs = json.load(open(os.path.join(this_dir, "seqs.json"), 'r'))
        seq = seqs[FLAGS.seq_id]
    logging.info(seq)
    summaries = []
    for config_id, num_trials in seq:
        if config_id == "None":
            config = get_config(FLAGS.__flags, {})
        else:
            configs_path = os.path.join(this_dir, "configs%s" % FLAGS.config_ext)
            config = get_config_from_file(FLAGS.__flags, configs_path, config_id)
        string = "{}\nConfig id {}, {} trials".format("="*80, config.config_id, num_trials)
        logging.info(string)
        print(string)
        summary = _main(config, num_trials)
        summaries.append(summary)

    string = "{}\nSUMMARY".format("="*80)
    logging.info(string)
    print(string)
    for summary in summaries:
        logging.info(summary)


if __name__ == "__main__":
    tf.app.run()
