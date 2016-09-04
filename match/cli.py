import os
from pprint import pprint

import tensorflow as tf

from match.main import main as m

flags = tf.app.flags

flags.DEFINE_string("model_name", "match", "Model name [match]")
flags.DEFINE_string("data_dir", "data/squad", "Data dir [data/squad]")
flags.DEFINE_integer("run_id", 0, "Run ID [0]")

flags.DEFINE_integer("batch_size", 128, "Batch size [128]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_integer("num_epochs", 50, "Total number of epochs for training [50]")
flags.DEFINE_integer("num_steps", 0, "Number of steps [0]")
flags.DEFINE_integer("eval_num_batches", 100, "eval num batches [100]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_integer("early_stop", 4, "early stop [4]")

flags.DEFINE_string("mode", "test", "train | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 1000, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay [0.9]")

flags.DEFINE_boolean("draft", False, "Draft for quick testing? [False]")

flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob [0.8]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_integer("char_filter_height", 5, "Char filter height [5]")
flags.DEFINE_float("wd", 0.0001, "Weight decay [0.001]")
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("dump_eval", False, "dump eval? [False]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_string("model", "1", "config 1 |2 [1]")

flags.DEFINE_integer("word_count_th", 10, "word count th [50]")
flags.DEFINE_integer("char_count_th", 50, "char count th [100]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")
flags.DEFINE_integer("ques_size_th", 32, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("answer_sent_size_th", 64, "answer sent size th [64]")


def main(_):
    config = flags.FLAGS

    config.out_dir = os.path.join("out", config.model_name, str(config.run_id).zfill(2))

    m(config)

if __name__ == "__main__":
    tf.app.run()
