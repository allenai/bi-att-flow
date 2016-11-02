import os

import tensorflow as tf

from basic_cnn.main import main as m

flags = tf.app.flags

flags.DEFINE_string("model_name", "basic_cnn", "Model name [basic]")
flags.DEFINE_string("data_dir", "data/cnn", "Data dir [data/cnn]")
flags.DEFINE_string("root_dir", "/Users/minjoons/data/cnn/questions", "root dir [~/data/cnn/questions]")
flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")

flags.DEFINE_integer("batch_size", 60, "Batch size [60]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_integer("num_epochs", 50, "Total number of epochs for training [50]")
flags.DEFINE_integer("num_steps", 20000, "Number of steps [20000]")
flags.DEFINE_integer("eval_num_batches", 100, "eval num batches [100]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_integer("early_stop", 4, "early stop [4]")

flags.DEFINE_string("mode", "test", "train | dev | test | forward [test]")
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
flags.DEFINE_float("wd", 0.0, "Weight decay [0.0]")
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("dump_eval", False, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_string("model", "2", "config 1 |2 [2]")
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")

flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 60, "sent size th [64]")
flags.DEFINE_integer("num_sents_th", 200, "num sents th [8]")
flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("finetune", False, "finetune? [False]")
flags.DEFINE_bool("feed_gt", False, "feed gt prev token during training [False]")
flags.DEFINE_bool("feed_hard", False, "feed hard argmax prev token during testing [False]")
flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
flags.DEFINE_bool("eval", True, "eval? [True]")
flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")

flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("load_path", "", "Load path []")
flags.DEFINE_string("shared_path", "", "Shared path []")
flags.DEFINE_string("device", "/cpu:0", "default device [/cpu:0]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus [1]")

flags.DEFINE_string("out_channel_dims", "100", "Out channel dims, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights, separated by commas [5]")

flags.DEFINE_bool("share_cnn_weights", True, "Share CNN weights [False]")
flags.DEFINE_bool("share_lstm_weights", True, "Share LSTM weights [True]")
flags.DEFINE_bool("two_prepro_layers", False, "Use two layers for preprocessing? [False]")
flags.DEFINE_bool("aug_att", False, "Augment attention layers with more features? [False]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("keep_prob", 1.0, "keep prob [1.0]")
flags.DEFINE_string("prev_mode", "a", "prev mode gy | y | a [a]")
flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_bool("sh", False, "use superhighway [False]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")
flags.DEFINE_bool("len_opt", False, "Length optimization? [False]")
flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")
flags.DEFINE_float("filter_ratio", 1.0, "filter ratio [1.0]")
flags.DEFINE_bool("bi", False, "bi-directional attention? [False]")
flags.DEFINE_integer("width", 5, "width around entity [5]")


def main(_):
    config = flags.FLAGS

    config.out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))

    m(config)

if __name__ == "__main__":
    tf.app.run()
