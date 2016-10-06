import os
from pprint import pprint

import tensorflow as tf

from basic.main import main as m

flags = tf.app.flags

flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("data_dir", "data/squad", "Data dir [data/squad]")
flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")

flags.DEFINE_integer("batch_size", 60, "Batch size [60]")
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
flags.DEFINE_integer("char_out_size", 100, "Char out size [100]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob [0.8]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_integer("char_filter_height", 5, "Char filter height [5]")
flags.DEFINE_float("wd", 0.0001, "Weight decay [0.001]")
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_string("model", "2", "config 1 |2 [2]")
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")

flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 60, "sent size th [64]")
flags.DEFINE_integer("num_sents_th", 8, "num sents th [8]")
flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("logit", "2l", "1l, 2l [2l]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("finetune", True, "finetune? [True]")
flags.DEFINE_bool("two_layers", False, "two layers [False]")
flags.DEFINE_bool("attention", False, "attention [False]")
flags.DEFINE_bool("greedy", False, "greedy [False]")
flags.DEFINE_bool("internal_attention", False, "internal attention [False]")
flags.DEFINE_bool("use_glove_for_unk", False, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", False, "consider as known if present in glove [False]")
flags.DEFINE_bool("eval", True, "eval? [True]")
flags.DEFINE_string("logit_func", "linear", "logit func [linear]")
flags.DEFINE_bool("traditional", True, "trad [True]")
flags.DEFINE_integer("num_gpus", 1, "num gpus to use [1]")

flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("load_path", "", "Load path []")
flags.DEFINE_string("shared_path", "", "Shared path []")

def main(_):
    config = flags.FLAGS

    config.out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))

    m(config)

if __name__ == "__main__":
    tf.app.run()
