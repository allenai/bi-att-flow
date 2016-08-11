import json
import os
import random
import itertools
import math

from my.utils import index


class DataSet(object):
    def __init__(self, data, data_type, shared=None):
        self.num_examples = len(next(iter(data.values())))
        self.data = data  # e.g. {'X': [0, 1, 2], 'Y': [2, 3, 4]}
        self.data_type = data_type
        self.shared = shared

    def get_batches(self, batch_size, num_batches=None, shuffle=False):
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        if num_batches is None:
            num_batches = num_batches_per_epoch
        num_epochs = int(math.ceil(num_batches / num_batches_per_epoch))

        idxs = itertools.chain.from_iterable(random.sample(range(self.num_examples), self.num_examples)
                                             if shuffle else range(self.num_examples)
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
            yield batch_ds


def load_metadata(config, data_type):
    metadata_path = os.path.join(config.data_dir, "metadata_{}.json".format(data_type))
    with open(metadata_path, 'r') as fh:
        metadata = json.load(fh)
        for key, val in metadata.items():
            config.__setattr__(key, val)
        return metadata


def read_data(config, data_type):
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
        data_set = DataSet(data, data_type)
        return data_set
