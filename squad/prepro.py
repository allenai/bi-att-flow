import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm


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
    # TODO : put more args here
    return parser.parse_args()


def prepro(args):
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
            if word in word_counter:
                vector = list(map(float, array[1:]))
                word2vec_dict[word] = vector
    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0):
    source_path = os.path.join(args.source_dir, "{}-v1.0-aug.json".format(data_type))
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    x, cx = [], []
    word_counter, char_counter = Counter(), Counter()

    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        for pi, para in enumerate(article['paragraphs']):
            xi = []
            for dep in para['deps']:
                if dep is None:
                    xi.append([])
                else:
                    xi.append([node[0] for node in dep[0]])
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            x.append(xi)
            cx.append(cxi)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            for qa in para['qas']:
                dep = qa['dep']
                qi = [] if dep is None else [node[0] for node in dep[0]]
                cqi = [list(qij) for qij in qi]
                for answer in qa['answers']:
                    yi0 = answer['answer_word_start'] or [0, 0]
                    yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    yi = [yi0, yi1]

                    q.append(qi)
                    cq.append(cqi)
                    y.append(yi)
                    rx.append(rxi)
                    rcx.append(rxi)
                    ids.append(qa['id'])
                    idxs.append(len(idxs))

                    for qij in qi:
                        word_counter[qij] += 1
                        for qijk in qij:
                            char_counter[qijk] += 1

    word2vec_dict = get_word2vec(args, word_counter)

    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'idxs': idxs, 'ids': ids}
    shared = {'x': x, 'cx': cx, 'word_counter': word_counter, 'char_counter': char_counter, 'word2vec': word2vec_dict}

    return data, shared


if __name__ == "__main__":
    main()