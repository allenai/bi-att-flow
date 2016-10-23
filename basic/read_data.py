import json
import os
import random
import itertools
import math

import numpy as np

from my.tensorflow import grouper
from my.utils import index


class DataSet(object):
    def __init__(self, data, data_type, shared=None, valid_idxs=None):
        total_num_examples = len(next(iter(data.values())))
        self.data = data  # e.g. {'X': [0, 1, 2], 'Y': [2, 3, 4]}
        self.data_type = data_type
        self.shared = shared
        self.valid_idxs = range(total_num_examples) if valid_idxs is None else valid_idxs
        self.num_examples = len(self.valid_idxs)

    def _sort_key(self, idx):
        rx = self.data['*x'][idx]
        x = self.shared['x'][rx[0]][rx[1]]
        return max(map(len, x))

    def get_batches(self, batch_size, num_batches=None, shuffle=False, cluster=False):
        """

        :param batch_size:
        :param num_batches:
        :param shuffle:
        :param cluster: cluster examples by their lengths; this might give performance boost (i.e. faster training).
        :return:
        """
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        if num_batches is None:
            num_batches = num_batches_per_epoch
        num_epochs = int(math.ceil(num_batches / num_batches_per_epoch))

        if shuffle:
            if cluster:
                sorted_idxs = sorted(self.valid_idxs, key=self._sort_key)
                sorted_grouped = lambda: list(grouper(sorted_idxs, batch_size))
                grouped = lambda: random.sample(sorted_grouped(), num_batches_per_epoch)
            else:
                random_grouped = lambda: list(grouper(random.sample(self.valid_idxs, len(self.valid_idxs)), batch_size))
                grouped = random_grouped
        else:
            raw_grouped = lambda: list(grouper(self.valid_idxs, batch_size))
            grouped = raw_grouped

        batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
        for _ in range(num_batches):
            batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
            batch_data = {}
            for key, val in self.data.items():
                if key.startswith('*'):
                    assert self.shared is not None
                    shared_key = key[1:]
                    batch_data[shared_key] = [index(self.shared[shared_key], val[idx]) for idx in batch_idxs]
                else:
                    batch_data[key] = list(map(val.__getitem__, batch_idxs))

            batch_ds = DataSet(batch_data, self.data_type, shared=self.shared)
            yield batch_idxs, batch_ds

    def get_multi_batches(self, batch_size, num_batches_per_step, num_steps=None, shuffle=False, cluster=False):
        num_batches = None if num_steps is None else num_batches_per_step * num_steps
        flat_batches = self.get_batches(batch_size, num_batches=num_batches, shuffle=shuffle, cluster=cluster)

        empty = next(self.get_batches(batch_size, num_batches=1))[1].get_empty()
        batches = grouper(flat_batches, num_batches_per_step, fillvalue=((), empty))
        return batches

    def get_empty(self):
        data = {key: [] for key in self.data}
        return DataSet(data, self.data_type, shared=self.shared)

    def __add__(self, other):
        data = {key: val + other.data[key] for key, val in self.data.items()}
        valid_idxs = list(self.valid_idxs) + [valid_idx + self.num_examples for valid_idx in other.valid_idxs]
        return DataSet(data, self.data_type, shared=self.shared, valid_idxs=valid_idxs)


class SquadDataSet(DataSet):
    def __init__(self, data, data_type, shared=None, valid_idxs=None):
        super(SquadDataSet, self).__init__(data, data_type, shared=shared, valid_idxs=valid_idxs)


def load_metadata(config, data_type):
    metadata_path = os.path.join(config.data_dir, "metadata_{}.json".format(data_type))
    with open(metadata_path, 'r') as fh:
        metadata = json.load(fh)
        for key, val in metadata.items():
            config.__setattr__(key, val)
        return metadata


def read_data(config, data_type, ref, data_filter=None):
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)

    num_examples = len(next(iter(data.values())))
    if data_filter is None:
        valid_idxs = range(num_examples)
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values):
            each = {key: val for key, val in zip(keys, vals)}
            mask.append(data_filter(each, shared))
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))

    shared_path = config.shared_path or os.path.join(config.out_dir, "shared.json")
    if not ref:
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        word_counter = shared['lower_word_counter'] if config.lower_word else shared['word_counter']
        char_counter = shared['char_counter']
        if config.finetune:
            shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th or (config.known_if_glove and word in word2vec_dict))}
        else:
            assert config.known_if_glove
            assert config.use_glove_for_unk
            shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th and word not in word2vec_dict)}
        shared['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter.items()
                                        if count > config.char_count_th)}
        NULL = "-NULL-"
        UNK = "-UNK-"
        shared['word2idx'][NULL] = 0
        shared['word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0
        shared['char2idx'][UNK] = 1
        json.dump({'word2idx': shared['word2idx'], 'char2idx': shared['char2idx']}, open(shared_path, 'w'))
    else:
        new_shared = json.load(open(shared_path, 'r'))
        for key, val in new_shared.items():
            shared[key] = val

    if config.use_glove_for_unk:
        # create new word2idx and word2vec
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
        shared['new_word2idx'] = new_word2idx_dict
        offset = len(shared['word2idx'])
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        shared['new_emb_mat'] = new_emb_mat

    data_set = DataSet(data, data_type, shared=shared, valid_idxs=valid_idxs)
    return data_set


def get_squad_data_filter(config):
    def data_filter(data_point, shared):
        assert shared is not None
        rx, rcx, q, cq, y = (data_point[key] for key in ('*x', '*cx', 'q', 'cq', 'y'))
        x, cx = shared['x'], shared['cx']
        if len(q) > config.ques_size_th:
            return False

        # x filter
        xi = x[rx[0]][rx[1]]
        if config.squash:
            for start, stop in y:
                stop_offset = sum(map(len, xi[:stop[0]]))
                if stop_offset + stop[1] > config.para_size_th:
                    return False
            return True

        if config.single:
            for start, stop in y:
                if start[0] != stop[0]:
                    return False

        if config.data_filter == 'max':
            for start, stop in y:
                    if stop[0] >= config.num_sents_th:
                        return False
                    if start[0] != stop[0]:
                        return False
                    if stop[1] >= config.sent_size_th:
                        return False
        elif config.data_filter == 'valid':
            if len(xi) > config.num_sents_th:
                return False
            if any(len(xij) > config.sent_size_th for xij in xi):
                return False
        elif config.data_filter == 'semi':
            """
            Only answer sentence needs to be valid.
            """
            for start, stop in y:
                if stop[0] >= config.num_sents_th:
                    return False
                if start[0] != start[0]:
                    return False
                if len(xi[start[0]]) > config.sent_size_th:
                    return False
        else:
            raise Exception()

        return True
    return data_filter


def update_config(config, data_sets):
    config.max_num_sents = 0
    config.max_sent_size = 0
    config.max_ques_size = 0
    config.max_word_size = 0
    config.max_para_size = 0
    for data_set in data_sets:
        data = data_set.data
        shared = data_set.shared
        for idx in data_set.valid_idxs:
            rx = data['*x'][idx]
            q = data['q'][idx]
            sents = shared['x'][rx[0]][rx[1]]
            config.max_para_size = max(config.max_para_size, sum(map(len, sents)))
            config.max_num_sents = max(config.max_num_sents, len(sents))
            config.max_sent_size = max(config.max_sent_size, max(map(len, sents)))
            config.max_word_size = max(config.max_word_size, max(len(word) for sent in sents for word in sent))
            if len(q) > 0:
                config.max_ques_size = max(config.max_ques_size, len(q))
                config.max_word_size = max(config.max_word_size, max(len(word) for word in q))

    if config.mode == 'train':
        config.max_num_sents = min(config.max_num_sents, config.num_sents_th)
        config.max_sent_size = min(config.max_sent_size, config.sent_size_th)
        config.max_para_size = min(config.max_para_size, config.para_size_th)

    config.max_word_size = min(config.max_word_size, config.word_size_th)

    config.char_vocab_size = len(data_sets[0].shared['char2idx'])
    config.word_emb_size = len(next(iter(data_sets[0].shared['word2vec'].values())))
    config.word_vocab_size = len(data_sets[0].shared['word2idx'])

    if config.single:
        config.max_num_sents = 1
    if config.squash:
        config.max_sent_size = config.max_para_size
        config.max_num_sents = 1
