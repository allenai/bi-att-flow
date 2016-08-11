import os
from pprint import pprint

import tensorflow as tf

from basic.main import main as m

flags = tf.app.flags

flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_integer("run_id", 0, "Run ID [0]")

flags.DEFINE_integer("batch_size", 32, "Batch size [32]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_integer("num_epochs", 50, "Total number of epochs for training [50]")
flags.DEFINE_integer("num_steps", None, "Number of steps [-1]")

flags.DEFINE_string("mode", "test", "train | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 10, "Log period [10]")
flags.DEFINE_integer("eval_period", 100, "Eval period [100]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")

flags.DEFINE_boolean("draft", False, "Draft for quick testing? [False]")


def main(_):
    config = flags.FLAGS

    config.data_dir = os.path.join("data", config.model_name)
    config.out_dir = os.path.join("out", config.model_name, str(config.run_id).zfill(2))

    pprint(config.__dict__, indent=2)
    m(config)

if __name__ == "__main__":
    tf.app.run()
