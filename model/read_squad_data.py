import json
import logging
import os

import numpy as np

from model.read_data import DataSet


class SharedDataSet(DataSet):
    def __init__(self, name, batch_size, data, shared, idxs, idx2id=None):
        super(SharedDataSet, self).__init__(name, batch_size, data, idxs, idx2id=idx2id)
        self.shared = shared

    def get_next_labeled_batch(self, partial=False):
        batch = super(SharedDataSet, self).get_next_labeled_batch(partial=partial)
        X = self.shared['X']
        refs = batch['R']
        batch['X'] = [X[i][j] for i, j in refs]
        return batch


def read_data(params, mode):
    batch_size = params.batch_size
    data_dir = params.data_dir

    mode2idxs_path = os.path.join(data_dir, "mode2idxs.json")
    data_path = os.path.join(data_dir, "batched.json")
    shared_path = os.path.join(data_dir, "shared.json")
    mode2idxs_dict = json.load(open(mode2idxs_path, 'r'))
    data = json.load(open(data_path, 'r'))
    shared = json.load(open(shared_path, 'r'))
    if mode not in mode2idxs_dict and mode == 'test':
        logging.warning("test data not found: using dev data instead.")
        idxs = mode2idxs_dict['dev']
    else:
        idxs = mode2idxs_dict[mode]
    data_set = SharedDataSet(mode, batch_size, data, shared, idxs)
    return data_set


def main():
    class Config:
        pass
    config = Config()
    config.batch_size = 2
    config.data_dir = "data/model/squad"
    data_set = read_data(config, "train")
    print(data_set.num_examples)
    print(data_set.get_next_labeled_batch())


if __name__ == "__main__":
    main()


