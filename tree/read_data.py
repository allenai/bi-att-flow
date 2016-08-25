import json
import os
import random
import itertools
import math

import nltk

from my.nltk_utils import load_compressed_tree
from my.utils import index


class DataSet(object):
    def __init__(self, data, data_type, shared=None, valid_idxs=None):
        total_num_examples = len(next(iter(data.values())))
        self.data = data  # e.g. {'X': [0, 1, 2], 'Y': [2, 3, 4]}
        self.data_type = data_type
        self.shared = shared
        self.valid_idxs = range(total_num_examples) if valid_idxs is None else valid_idxs
        self.num_examples = len(self.valid_idxs)

    def get_batches(self, batch_size, num_batches=None, shuffle=False):
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        if num_batches is None:
            num_batches = num_batches_per_epoch
        num_epochs = int(math.ceil(num_batches / num_batches_per_epoch))

        idxs = itertools.chain.from_iterable(random.sample(self.valid_idxs, len(self.valid_idxs))
                                             if shuffle else self.valid_idxs
                                             for _ in range(num_epochs))
        for _ in range(num_batches):
            batch_idxs = tuple(itertools.islice(idxs, batch_size))
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

    shared_path = os.path.join(config.out_dir, "shared.json")
    if not ref:
        word_counter = shared['lower_word_counter'] if config.lower_word else shared['word_counter']
        char_counter = shared['char_counter']
        pos_counter = shared['pos_counter']
        shared['word2idx'] = {word: idx + 2 for idx, word in
                              enumerate(word for word, count in word_counter.items()
                                        if count > config.word_count_th)}
        shared['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter.items()
                                        if count > config.char_count_th)}
        shared['pos2idx'] = {pos: idx + 2 for idx, pos in enumerate(pos_counter.keys())}
        NULL = "-NULL-"
        UNK = "-UNK-"
        shared['word2idx'][NULL] = 0
        shared['word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0
        shared['char2idx'][UNK] = 1
        shared['pos2idx'][NULL] = 0
        shared['pos2idx'][UNK] = 1
        json.dump({'word2idx': shared['word2idx'], 'char2idx': shared['char2idx'],
                   'pos2idx': shared['pos2idx']}, open(shared_path, 'w'))
    else:
        new_shared = json.load(open(shared_path, 'r'))
        for key, val in new_shared.items():
            shared[key] = val

    data_set = DataSet(data, data_type, shared=shared, valid_idxs=valid_idxs)
    return data_set


def get_squad_data_filter(config):
    def data_filter(data_point, shared):
        assert shared is not None
        rx, rcx, q, cq, y  = (data_point[key] for key in ('*x', '*cx', 'q', 'cq', 'y'))
        x, cx, stx = shared['x'], shared['cx'], shared['stx']
        if len(q) > config.ques_size_th:
            return False
        xi = x[rx[0]][rx[1]]
        if len(xi) > config.num_sents_th:
            return False
        if any(len(xij) > config.sent_size_th for xij in xi):
            return False
        stxi = stx[rx[0]][rx[1]]
        if any(nltk.tree.Tree.fromstring(s).height() > config.tree_height_th for s in stxi):
            return False
        return True
    return data_filter


def update_config(config, data_sets):
    config.max_num_sents = 0
    config.max_sent_size = 0
    config.max_ques_size = 0
    config.max_word_size = 0
    config.max_tree_height = 0
    for data_set in data_sets:
        data = data_set.data
        shared = data_set.shared
        for idx in data_set.valid_idxs:
            rx = data['*x'][idx]
            q = data['q'][idx]
            sents = shared['x'][rx[0]][rx[1]]
            trees = map(nltk.tree.Tree.fromstring, shared['stx'][rx[0]][rx[1]])
            config.max_tree_height = max(config.max_tree_height, max(tree.height() for tree in trees))
            config.max_num_sents = max(config.max_num_sents, len(sents))
            config.max_sent_size = max(config.max_sent_size, max(map(len, sents)))
            config.max_word_size = max(config.max_word_size, max(len(word) for sent in sents for word in sent))
            if len(q) > 0:
                config.max_ques_size = max(config.max_ques_size, len(q))
                config.max_word_size = max(config.max_word_size, max(len(word) for word in q))

    config.max_word_size = min(config.max_word_size, config.word_size_th)

    config.char_vocab_size = len(data_sets[0].shared['char2idx'])
    config.word_emb_size = len(next(iter(data_sets[0].shared['word2vec'].values())))
    config.word_vocab_size = len(data_sets[0].shared['word2idx'])
    config.pos_vocab_size = len(data_sets[0].shared['pos2idx'])
