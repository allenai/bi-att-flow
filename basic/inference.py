import os
from pprint import pprint

import tensorflow as tf
import numpy as np

from basic.evaluator import ForwardEvaluator
from basic.graph_handler import GraphHandler
from basic.model import get_multi_gpu_models

from basic.main import set_dirs
from basic.read_data import read_data, update_config, DataSet
from squad.prepro import prepro_single_question_with_context

flags = tf.app.flags

# Names and directories
flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("dataset", "dev", "Dataset [test]")
#flags.DEFINE_string("data_dir", "/Applications/MAMP/htdocs/bi-att-flow/data/squad", "Data dir [data/squad]")
flags.DEFINE_string("data_dir", "", "Data dir [data/squad]")
flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("eval_path", "data/squad", "Eval path []")
# flags.DEFINE_string("load_path", "", "Load path []")
# flags.DEFINE_string("load_path", "/Applications/MAMP/htdocs/bi-att-flow/save/40/save", "Load path []")
flags.DEFINE_string("load_path", "", "Load path []")
#flags.DEFINE_string("load_path", "out/basic/00/save/basic-20000", "Load path []")
# "$root_dir/$num/shared.json"
#flags.DEFINE_string("shared_path", "/Applications/MAMP/htdocs/bi-att-flow/save/40/shared.json", "Shared path []")
flags.DEFINE_string("shared_path", "", "Shared path []")
flags.DEFINE_integer("eval_num_batches", 0, "eval num batches [100]")

# Device placement
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

# Essential training and test options
#flags.DEFINE_string("mode", "test", "trains | test | forward [test]")
flags.DEFINE_string("mode", "forward", "trains | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
flags.DEFINE_bool("eval", True, "eval? [True]")

# Training / test parameters
flags.DEFINE_integer("batch_size", 1, "Batch size [60]")
flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
flags.DEFINE_integer("num_steps", 20000, "Number of steps [20000]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")
flags.DEFINE_bool("highway", True, "Use highway? [True]")
flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")

# Optimizations
flags.DEFINE_bool("cluster", True, "Cluster data for faster training [False]")
flags.DEFINE_bool("len_opt", True, "Length optimization? [False]")
flags.DEFINE_bool("cpu_opt", True, "CPU optimization? GPU computation can be slower [False]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 1000, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

# Thresholds for speed and less memory usage
flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 400, "sent size th [64]")
flags.DEFINE_integer("num_sents_th", 8, "num sents th [8]")
flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

# Advanced training options
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")

# Ablation options
flags.DEFINE_bool("use_char_emb", True, "use char emb? [True]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")
flags.DEFINE_bool("dynamic_att", False, "Dynamic attention [False]")


class Inference(object):
    def __init__(self):
        self.config = flags.FLAGS

        # Set directories for temporary files
        self.config.out_dir = os.path.join(self.config.out_base_dir, self.config.model_name, str(self.config.run_id).zfill(2))
        set_dirs(self.config)

        # TODO: Can we refactor this?
        self.config.max_sent_size = self.config.sent_size_th
        self.config.max_num_sents = self.config.num_sents_th
        self.config.max_ques_size = self.config.ques_size_th
        self.config.max_word_size = self.config.word_size_th
        self.config.max_para_size = self.config.para_size_th

        # Read trained dataset and update word embedding matrix.
        # We only really need this to get word2idx, idx2word
        # and the word embedding matrix
        self.data = read_data(self.config, self.config.dataset, True)
        self.update_embedding_matrix()

        # Get the models and the evaluator to get predictions
        self.model = get_multi_gpu_models(self.config)[0]
        self.evaluator = ForwardEvaluator(
            config=self.config,
            model=self.model,
            tensor_dict=models.tensor_dict if self.config.vis else None
        )

        # Initialize TF session and graph handler
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.graph_handler = GraphHandler(self.config, self.model)
        self.graph_handler.initialize(self.sess)

        pprint(flags.FLAGS.__dict__, indent=2)

    def update_embedding_matrix(self):
        update_config(self.config, [self.data])
        if self.config.use_glove_for_unk:
            # Get the word2vec or lower_word2vec
            word2vec_dict = self.data.shared['lower_word2vec'] if self.config.lower_word else self.data.shared['word2vec']
            new_word2idx_dict = self.data.shared['new_word2idx']
            idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
            new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
            self.config.new_emb_mat = new_emb_mat

    def predict(self, context, question):

        batch_data_prepro = prepro_single_question_with_context(context, question)
        pprint(batch_data_prepro)

        batch_data = ((0,), DataSet(
            data=batch_data_prepro,
            data_type=self.config.dataset,
            shared=self.data.shared
        ))
        prediction = self.evaluator.get_evaluation(self.sess, batch_data)

        p = prediction.id2answer_dict[
            list(prediction.id2answer_dict.keys())[0]
        ]
        s = prediction.id2answer_dict['scores'][
            list(prediction.id2answer_dict['scores'].keys())[0]
        ]

        return p, s

if __name__ == "__main__":

    inference = Inference()
    context = 'More than 26,000 square kilometres (10,000 sq mi) of Victorian farmland are sown for grain, mostly in the state west. More than 50% of this area is sown for wheat, 33% is sown for barley and 7% is sown for oats. A further 6,000 square kilometres (2,300 sq mi) is sown for hay. In 2003–04, Victorian farmers produced more than 3 million tonnes of wheat and 2 million tonnes of barley. Victorian farms produce nearly 90% of Australian pears and third of apples. It is also a leader in stone fruit production. The main vegetable crops include asparagus, broccoli, carrots, potatoes and tomatoes. Last year, 121,200 tonnes of pears and 270,000 tonnes of tomatoes were produced.'
    question = 'What percentage of farmland grows wheat? '
    print(inference.predict(context, question))