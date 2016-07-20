import json
import logging
import os

import numpy as np

from my.utils import _index

NUM = "NUM"


class DataSet(object):
    def __init__(self, data, batch_size, shared=None, name='default', idxs=None, idx2id=None):
        self.name = name
        self.num_epochs_completed = 0
        self.idx_in_epoch = 0
        self.batch_size = batch_size
        self.data = data
        self.num_examples = len(next(iter(data.values())))
        if idxs is None:
            idxs = list(range(self.num_examples))
        self.init_idxs = idxs
        self.idxs = idxs
        if idx2id is None:
            idx2id = {idx: idx for idx in idxs}
        self.idx2id = idx2id
        self.num_full_batches = int(self.num_examples / self.batch_size)
        self.num_all_batches = self.num_full_batches + int(self.num_examples % self.batch_size > 0)
        self.shared = shared

    def get_num_batches(self, partial=False):
        return self.num_all_batches if partial else self.num_full_batches

    def get_batch_idxs(self, partial=False):
        assert self.has_next_batch(partial=partial), "End of data, reset required."
        from_, to = self.idx_in_epoch, self.idx_in_epoch + self.batch_size
        if partial and to > self.num_examples:
            to = self.num_examples
        cur_idxs = self.idxs[from_:to]
        return cur_idxs

    def get_next_batch(self, partial=False):
        cur_idxs = self.get_batch_idxs(partial=partial)
        batch = {key: [each[i] for i in cur_idxs] for key, each in self.data.items()}
        assert NUM not in batch, "Variable name '{}' is reserved.".format(NUM)
        batch[NUM] = len(cur_idxs)

        self.idx_in_epoch += len(cur_idxs)

        # handle shared data
        new_ = {}
        for key, data in batch.items():
            if key.startswith("*"):
                new_key = key[1:]
                new_[new_key] = [_index(self.shared, ref) for ref in data]
        for key, data in new_.items():
            batch[key] = data

        return batch

    def has_next_batch(self, partial=False):
        if partial:
            return self.idx_in_epoch < self.num_examples
        return self.idx_in_epoch + self.batch_size <= self.num_examples

    def complete_epoch(self, shuffle=True):
        self.reset(shuffle=shuffle)
        self.num_epochs_completed += 1

    def reset(self, shuffle=True):
        self.idx_in_epoch = 0
        if shuffle:
            np.random.shuffle(self.idxs)
        else:
            self.idxs = self.init_idxs


def read_data(params, mode):
    logging.info("loading {} data ... ".format(mode))
    batch_size = params.batch_size
    data_dir = params.data_dir

    data_path = os.path.join(data_dir, "{}_data.json".format(mode))
    data = json.load(open(data_path, 'r'))
    shared_path = os.path.join(data_dir, "{}_shared.json".format(mode))
    shared = json.load(open(shared_path, 'r'))
    data_set = DataSet(data, batch_size, shared=shared, name=mode)
    return data_set
