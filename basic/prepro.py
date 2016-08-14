import argparse
import json
import os
import itertools
from collections import Counter

import numpy as np
from tqdm import tqdm
import nltk

from nltk_utils import set_span, tree_contains_span, find_max_f1_span

NULL = "<NULL>"
UNK = "<UNK>"


def bool_(arg):
    if arg == 'True':
        return True
    elif arg == 'False':
        return False
    raise Exception()


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/squad"
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=target_dir)
    parser.add_argument("--min_word_count", default=100, type=int)
    parser.add_argument("--min_char_count", default=500, type=int)
    parser.add_argument("--max_word_size", default=16, type=int)
    parser.add_argument("--max_num_sents", default=16, type=int)
    parser.add_argument("--max_sent_size", default=64, type=int)
    parser.add_argument("--debug", default=False, type=bool_)
    parser.add_argument("--train_ratio", default=0.9, type=int)
    # TODO : put more args here
    return parser.parse_args()


def get_data(args, data_path, is_train):
    arg_max_num_sents = args.max_num_sents if is_train else 9999
    arg_max_sent_size = args.max_sent_size if is_train else 9999

    with open(data_path, 'r') as fh:
        d = json.load(fh)
        size = sum(len(article['paragraphs']) for article in d['data'])
        pbar = tqdm(range(size))
        f1s = []
        max_num_sents = 0
        max_sent_size = 0
        max_num_words = 0
        max_ques_size = 0
        max_ques_word_size = 0
        max_sent_word_size = 0

        rx, q, y = [], [], []
        cq = []
        x = []
        cx = []

        word_counter = Counter()
        char_counter = Counter()
        invalid_stop_idx_counter = 0

        for ai, article in enumerate(d['data']):
            x_a, cx_a = [], []
            x.append(x_a)
            cx.append(cx_a)
            for pi, para in enumerate(article['paragraphs']):
                ref = [ai, pi]
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

                context_words = [[each[0][:args.max_word_size] for each in nodes] for nodes in context_nodes]
                context_chars = [[list(word) for word in sent] for sent in context_words]
                x_a.append(context_words)
                cx_a.append(context_chars)
                word_counter.update(word for sent in context_words for word in sent)
                char_counter.update(char for sent in context_words for word in sent for char in word)

                if len(context_nodes) > arg_max_num_sents:
                    continue
                if max(map(len, context_nodes)) > arg_max_sent_size:
                    continue
                max_num_sents = max(max_num_sents, len(context_nodes))
                max_sent_size = max(max_sent_size, max(map(len, context_nodes)))
                max_num_words = max(max_num_words, sum(map(len, context_nodes)))
                max_sent_word_size = max(max_sent_word_size, max(len(word) for sent in context_words for word in sent))
                consts = para['context_const']
                # context_words, context_tags, context_starts, context_stops = zip(*context_nodes)
                for qa in para['qas']:
                    question = qa['question']
                    question_dep = qa['question_dep']
                    if question_dep is None:
                        print("unparsed question (ignoring): {}".format(question))
                    question_words = [] if question_dep is None else [each[0][:args.max_word_size] for each in question_dep[0]]
                    question_chars = [[]] if question_dep is None else [list(word) for word in question_words]
                    max_ques_size = max(max_ques_size, len(question_words))
                    max_ques_word_size = max(max_ques_word_size, max(map(len, question_chars)))
                    bs = []
                    for answer in qa['answers'][:1]:  # Fix this to use all answers!
                        start_idx = answer['start_idx']
                        stop_idx = answer['stop_idx']
                        # If span is extended further than the sent length
                        if stop_idx[1] > len(context_words[stop_idx[0]]):
                            invalid_stop_idx_counter += 1
                        full_span = [start_idx, stop_idx]
                        support_tree = nltk.tree.Tree.fromstring(consts[start_idx[0]])
                        span = (start_idx[1], stop_idx[1])
                        set_span(support_tree)
                        b = int(tree_contains_span(support_tree, span))
                        bs.append(b)

                        max_span, f1 = find_max_f1_span(support_tree, span)
                        f1s.append(f1)

                        rx.append(ref)
                        q.append(question_words)
                        cq.append(question_chars)
                        y.append(full_span)
            if args.debug:
                break
        print("num invalid stop idx: {}".format(invalid_stop_idx_counter))
        print("average f1: {}".format(np.mean(f1s)))
        print("max sent size: {}".format(max_sent_size))
        print("max num words: {}".format(max_num_words))
        print("max num sents: {}".format(max_num_sents))
        print("max ques size: {}".format(max_ques_size))
        print("max sent word size: {}".format(max_sent_word_size))
        print("max ques word size: {}".format(max_ques_word_size))

        wv = {word: i+2 for i, word in enumerate(word for word, count in word_counter.items() if count >= args.min_word_count)}
        cv = {char: i+2 for i, char in enumerate(char for char, count in char_counter.items() if count >= args.min_char_count)}
        assert NULL not in wv
        assert UNK not in wv
        wv[NULL] = 0
        wv[UNK] = 1
        cv[NULL] = 0
        cv[UNK] = 1

        metadata = {'max_sent_size': max_sent_size,
                    'max_num_words': max_num_words,
                    'max_num_sents': max_num_sents,
                    'max_ques_size': max_ques_size,
                    'max_sent_word_size': max_sent_word_size,
                    'max_ques_word_size': max_ques_word_size,
                    'word_vocab_size': len(wv),
                    'char_vocab_size': len(cv)}
        data = {'*x': rx, 'cq': cq, 'q': q, 'y': y}
        shared = {'x': x, 'cx': cx, 'wv': wv, 'cv': cv}
        return data, shared, metadata


def recursive_replace(l, v):
    if isinstance(l, str):
        if l in v:
            return v[l]
        else:
            return 1
    return [recursive_replace(each, v) for each in l]


def apply(data, shared, wv, cv):
    data = {'*x': data['*x'], 'cq': recursive_replace(data['cq'], cv), 'q': recursive_replace(data['q'], wv), 'y': data['y']}
    shared = {'x': recursive_replace(shared['x'], wv), 'cx': recursive_replace(shared['cx'], cv), 'wv': shared['wv'], 'cv': shared['cv']}
    return data, shared


def split(data, ratio):
    idx = int(ratio * len(next(iter(data.values()))))
    train_data = {key: val[:idx] for key, val in data.items()}
    dev_data = {key: val[idx:] for key, val in data.items()}
    return train_data, dev_data


def prepro(args):
    train_data_path = os.path.join(args.source_dir, "train-v1.0-aug.json")
    test_data_path = os.path.join(args.source_dir, "dev-v1.0-aug.json")
    data_train, shared_train, metadata_train = get_data(args, train_data_path, True)
    data_test, shared_test, metadata_test = get_data(args, test_data_path, False)

    wv = shared_train['wv']
    cv = shared_train['cv']
    data_train, shared_train = apply(data_train, shared_train, wv, cv)
    data_test, shared_test = apply(data_test, shared_test, wv, cv)
    data_train, data_dev = split(data_train, args.train_ratio)

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    def save(data, shared, metadata, data_type):
        out_data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
        out_shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
        metadata_path = os.path.join(args.target_dir, "metadata_{}.json".format(data_type))
        with open(out_data_path, 'w') as fh:
            json.dump(data, fh)
        with open(out_shared_path, 'w') as fh:
            json.dump(shared, fh)
        with open(metadata_path, 'w') as fh:
            json.dump(metadata, fh)

    save(data_train, shared_train, metadata_train, 'train')
    save(data_dev, shared_train, metadata_train, 'dev')
    save(data_test, shared_test, metadata_test, 'test')


def main():
    args = get_args()
    prepro(args)

if __name__ == "__main__":
    main()
