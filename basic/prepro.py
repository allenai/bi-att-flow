import argparse
import json
import os
import itertools

import numpy as np
from tqdm import tqdm
import nltk

from nltk_utils import set_span, tree_contains_span


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/basic"
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=target_dir)
    parser.add_argument("--size", default=1000, type=int)
    parser.add_argument("--std", default=0.5, type=float)
    # TODO : put more args here
    return parser.parse_args()


def get_data(data_path):
    with open(data_path, 'r') as fh:
        d = json.load(fh)
        size = sum(len(article['paragraphs']) for article in d['data'])
        pbar = tqdm(range(size))
        pos_counter = 0
        neg_counter = 0
        for article in d['data']:
            for para in article['paragraphs']:
                pbar.update(1)
                # context = para['context']
                context_nodes, context_edges = [], []
                for each in para['context_dep']:
                    if each is None:
                        # ignores as non-existent
                        context_nodes.append([])
                        context_edges.append([])
                    else:
                        nodes, edges = each
                        context_nodes.append(nodes)
                        context_edges.append(edges)
                consts = para['context_const']
                # context_words, context_tags, context_starts, context_stops = zip(*context_nodes)
                for qa in para['qas']:
                    # question = qa['question']
                    question_dep = qa['question_dep']
                    bs = []
                    for answer in qa['answers']:
                        start_idx = answer['start_idx']
                        stop_idx = answer['stop_idx']
                        support_tree = nltk.tree.Tree.fromstring(consts[start_idx[0]])
                        span = (start_idx[1], stop_idx[1])
                        set_span(support_tree)
                        b = int(tree_contains_span(support_tree, span))
                        bs.append(b)
                    b = max(bs)
                    pos_counter += b
                    neg_counter += 1 - b
        total = pos_counter + neg_counter
        print(pos_counter, neg_counter, pos_counter / total)

        return None, None


def prepro(args):
    train_data_path = os.path.join(args.source_dir, "train-v1.0-aug.json")
    dev_data_path = os.path.join(args.source_dir, "dev-v1.0-aug.json")
    data_train, shared_train = get_data(train_data_path)
    data_dev, shared_dev = get_data(dev_data_path)

def main():
    args = get_args()
    prepro(args)

if __name__ == "__main__":
    main()
