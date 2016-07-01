import argparse
import json
import os
import numpy as np
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "model")
    target_dir = "data/model"
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=target_dir)
    # TODO : put more args here
    return parser.parse_args()


def prepro(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    # TODO : put something here; Fake data shown
    size = 1000
    std = 0.1
    Y = [0] * size + [1] * size
    X = np.random.normal(0, std, size).tolist() + np.random.normal(1, std, size).tolist()
    idxs = np.random.permutation(size * 2).tolist()
    train_idxs = idxs[:int(size * 0.6)]
    dev_idxs = idxs[int(size * 0.6):int(size * 0.7)]
    test_idxs = idxs[int(size * 0.7):]

    data = {'X': X, 'Y': Y}
    mode2idxs_dict = {'train': train_idxs, 'dev': dev_idxs, 'test': test_idxs}
    _save(mode2idxs_dict, data, target_dir)


def _save(mode2idxs_dict, data, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    mode2idxs_path = os.path.join(target_dir, "mode2idxs.json")
    metadata_path = os.path.join(target_dir, "metadata.json")
    data_path = os.path.join(target_dir, "data.json")

    X, Y = data['X'], data['Y']
    metadata = {'num_classes': len(set(Y))}

    with open(mode2idxs_path, 'w') as fh:
        json.dump(mode2idxs_dict, fh)
    with open(metadata_path, 'w') as fh:
        json.dump(metadata, fh)
    with open(data_path, 'w') as fh:
        json.dump(data, fh)


def main():
    args = get_args()
    prepro(args)


if __name__ == "__main__":
    main()
