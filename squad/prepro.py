import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

import nltk
from tqdm import tqdm

from my.nltk_utils import load_compressed_tree


def bool_(arg):
    if arg == 'True':
        return True
    elif arg == 'False':
        return False
    raise Exception()


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/squad"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=target_dir)
    parser.add_argument("--debug", default=False, type=bool_)
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--full_train", default=False, type=bool_)
    # TODO : put more args here
    return parser.parse_args()


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.full_train:
        data_train, shared_train = prepro_each(args, 'train')
        data_dev, shared_dev = prepro_each(args, 'dev')
    else:
        data_train, shared_train = prepro_each(args, 'train', 0.0, args.train_ratio)
        data_dev, shared_dev = prepro_each(args, 'train', args.train_ratio, 1.0)
    data_test, shared_test = prepro_each(args, 'dev')

    print("saving ...")
    save(args, data_train, shared_train, 'train')
    save(args, data_dev, shared_dev, 'dev')
    save(args, data_test, shared_test, 'test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0):
    source_path = os.path.join(args.source_dir, "{}-v1.0-aug.json".format(data_type))
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    x, cx, tx, stx = [], [], [], []
    answerss = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    pos_counter = Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp, txp, stxp = [], [], [], []
        x.append(xp)
        cx.append(cxp)
        tx.append(txp)
        stx.append(stxp)
        for pi, para in enumerate(article['paragraphs']):
            xi = []
            for dep in para['deps']:
                if dep is None:
                    xi.append([])
                else:
                    xi.append([node[0] for node in dep[0]])
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            txp.append(para['consts'])
            stxp.append([str(load_compressed_tree(s)) for s in para['consts']])
            trees = map(nltk.tree.Tree.fromstring, para['consts'])
            for tree in trees:
                for subtree in tree.subtrees():
                    pos_counter[subtree.label()] += 1

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            for qa in para['qas']:
                dep = qa['dep']
                qi = [] if dep is None else [node[0] for node in dep[0]]
                cqi = [list(qij) for qij in qi]
                yi = []
                answers = []
                for answer in qa['answers']:
                    answers.append(answer['text'])
                    yi0 = answer['answer_word_start'] or [0, 0]
                    yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    yi.append([yi0, yi1])

                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1

                q.append(qi)
                cq.append(cqi)
                y.append(yi)
                rx.append(rxi)
                rcx.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                answerss.append(answers)

            if args.debug:
                break

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, '*tx': rx, '*stx': rx,
            'idxs': idxs, 'ids': ids, 'answerss': answerss}
    shared = {'x': x, 'cx': cx, 'tx': tx, 'stx': stx,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict, 'pos_counter': pos_counter}

    return data, shared


if __name__ == "__main__":
    main()