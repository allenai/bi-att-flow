import os
from pprint import pprint

import tensorflow as tf

from basic.main import main as m

flags = tf.app.flags

flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("data_dir", "data/squad", "Data dir [data/squad]")
flags.DEFINE_integer("run_id", 0, "Run ID [0]")

flags.DEFINE_integer("batch_size", 32, "Batch size [32]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_integer("num_epochs", 50, "Total number of epochs for training [50]")
flags.DEFINE_integer("num_steps", None, "Number of steps [-1]")

flags.DEFINE_string("mode", "test", "train | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 10, "Log period [10]")
flags.DEFINE_integer("eval_period", 5000, "Eval period [5000]")
flags.DEFINE_integer("save_period", 5000, "Save Period [5000]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay [0.9]")

flags.DEFINE_boolean("draft", False, "Draft for quick testing? [False]")

flags.DEFINE_integer("hidden_size", 64, "Hidden size [100]")
flags.DEFINE_float("input_keep_prob", 0.5, "Input keep prob [0.5]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_integer("char_filter_height", 4, "Char filter height [4]")
flags.DEFINE_float("wd", 0.001, "Weight decay [0.001]")
flags.DEFINE_bool("pool_rnn", False, "Pool RNN [False]")
flags.DEFINE_bool("tanh_dot", False, "Tanh Dot [ False]")


def main(_):
    config = flags.FLAGS

    config.out_dir = os.path.join("out", config.model_name, str(config.run_id).zfill(2))

    m(config)

if __name__ == "__main__":
    tf.app.run()
