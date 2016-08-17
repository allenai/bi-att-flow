import json
import os
import random
import itertools
import math

from my.utils import index


class DataSet(object):
    def __init__(self, data, data_type, shared=None, valid_idxs=None):
        self.num_examples = len(next(iter(data.values())))
        self.data = data  # e.g. {'X': [0, 1, 2], 'Y': [2, 3, 4]}
        self.data_type = data_type
        self.shared = shared
        self.valid_idxs = range(self.num_examples) if valid_idxs is None else valid_idxs

    def get_batches(self, batch_size, num_batches=None, shuffle=False, data_filter=None):
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


def load_metadata(config, data_type):
    metadata_path = os.path.join(config.data_dir, "metadata_{}.json".format(data_type))
    with open(metadata_path, 'r') as fh:
        metadata = json.load(fh)
        for key, val in metadata.items():
            config.__setattr__(key, val)
        return metadata


def read_data(config, data_type, data_filter=None):
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)

    if data_filter is None:
        valid_idxs = len(next(iter(data)))
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values):
            each = {key: val for key, val in zip(keys, vals)}
            mask.append(data_filter(each, shared))
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {} examples from {}".format(len(valid_idxs), data_type))

    data_set = DataSet(data, data_type, shared=shared, valid_idxs=valid_idxs)
    return data_set


def get_squad_data_filter(config):
    config.max_num_sents = config.num_sents_th
    config.max_ques_size = config.ques_size_th
    config.max_sent_size = config.sent_size_th

    def data_filter(data_point, shared):
        assert shared is not None
        rx, rcx, q, cq, y = (data_point[key] for key in ('*x', '*cx', 'q', 'cq', 'y'))
        x, cx = shared['x'], shared['cx']
        if len(q) > config.ques_size_th:
            return False
        xi = x[rx[0]][rx[1]]
        if len(xi) > config.num_sents_th:
            return False
        if any(len(xij) > config.sent_size_th for xij in xi):
            return False
        return True
    return data_filter

