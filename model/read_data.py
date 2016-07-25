import json
import logging
import os

import numpy as np

NUM = "NUM"

class DataSet(object):
    def __init__(self, name, batch_size, data, idxs, idx2id=None, shuffle=True):
        self.name = name
        self.num_epochs_completed = 0
        self.idx_in_epoch = 0
        self.batch_size = batch_size
        self.data = data
        self.idxs = idxs
        if idx2id is None:
            idx2id = {idx: idx for idx in idxs}
        self.idx2id = idx2id
        self.num_examples = len(idxs)
        self.num_full_batches = int(self.num_examples / self.batch_size)
        self.num_all_batches = self.num_full_batches + int(self.num_examples % self.batch_size > 0)
        self.shuffle = shuffle
        self.reset()

    def get_num_batches(self, partial=False):
        return self.num_all_batches if partial else self.num_full_batches

    def get_batch_idxs(self, partial=False):
        assert self.has_next_batch(partial=partial), "End of data, reset required."
        from_, to = self.idx_in_epoch, self.idx_in_epoch + self.batch_size
        if partial and to > self.num_examples:
            to = self.num_examples
        cur_idxs = self.idxs[from_:to]
        return cur_idxs

    def get_next_labeled_batch(self, partial=False):
        cur_idxs = self.get_batch_idxs(partial=partial)
        batch = {key: [each[i] for i in cur_idxs] for key, each in self.data.items()}
        assert NUM not in batch, "Variable name '{}' is reserved.".format(NUM)
        batch[NUM] = len(cur_idxs)

        self.idx_in_epoch += len(cur_idxs)
        return batch

    def has_next_batch(self, partial=False):
        if partial:
            return self.idx_in_epoch < self.num_examples
        return self.idx_in_epoch + self.batch_size <= self.num_examples

    def complete_epoch(self):
        self.reset()
        self.num_epochs_completed += 1

    def reset(self):
        self.idx_in_epoch = 0
        # For debugging purpose, shuffle can be turned off
        if self.shuffle:
            np.random.shuffle(self.idxs)


def read_data(params, mode):
    logging.info("loading {} data ... ".format(mode))
    batch_size = params.batch_size
    data_dir = params.data_dir

    mode2idxs_path = os.path.join(data_dir, "mode2idxs.json")
    data_path = os.path.join(data_dir, "data.json")
    mode2idxs_dict = json.load(open(mode2idxs_path, 'r'))
    data = json.load(open(data_path, 'r'))
    idxs = mode2idxs_dict[mode]
    data_set = DataSet(mode, batch_size, data, idxs)
    return data_set
