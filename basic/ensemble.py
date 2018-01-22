import argparse
import functools
import gzip
import json
import pickle
from collections import defaultdict
from operator import mul

from tqdm import tqdm
from squad.utils import get_phrase, get_best_span, get_span_score_pairs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('-o', '--out', default='ensemble.json')
    parser.add_argument("--data_path", default="data/squad/data_test.json")
    parser.add_argument("--shared_path", default="data/squad/shared_test.json")
    args = parser.parse_args()
    return args


def ensemble(args):
    e_list = []
    for path in tqdm(args.paths):
        with gzip.open(path, 'r') as fh:
            e = pickle.load(fh)
            e_list.append(e)

    with open(args.data_path, 'r') as fh:
        data = json.load(fh)

    with open(args.shared_path, 'r') as fh:
        shared = json.load(fh)

    out = {}
    for idx, (id_, rx) in tqdm(enumerate(zip(data['ids'], data['*x'])), total=len(e['yp'])):
        if idx >= len(e['yp']):
            # for debugging purpose
            break
        context = shared['p'][rx[0]][rx[1]]
        wordss = shared['x'][rx[0]][rx[1]]
        yp_list = [e['yp'][idx] for e in e_list]
        yp2_list = [e['yp2'][idx] for e in e_list]
        answer = ensemble4(context, wordss, yp_list, yp2_list)
        out[id_] = answer

    with open(args.out, 'w') as fh:
        json.dump(out, fh)


def ensemble1(context, wordss, y1_list, y2_list):
    """

    :param context: Original context
    :param wordss: tokenized words (nested 2D list)
    :param y1_list: list of start index probs (each element corresponds to probs form single model)
    :param y2_list: list of stop index probs
    :return:
    """
    sum_y1 = combine_y_list(y1_list)
    sum_y2 = combine_y_list(y2_list)
    span, score = get_best_span(sum_y1, sum_y2)
    return get_phrase(context, wordss, span)


def ensemble2(context, wordss, y1_list, y2_list):
    start_dict = defaultdict(float)
    stop_dict = defaultdict(float)
    for y1, y2 in zip(y1_list, y2_list):
        span, score = get_best_span(y1, y2)
        start_dict[span[0]] += y1[span[0][0]][span[0][1]]
        stop_dict[span[1]] += y2[span[1][0]][span[1][1]]
    start = max(start_dict.items(), key=lambda pair: pair[1])[0]
    stop = max(stop_dict.items(), key=lambda pair: pair[1])[0]
    best_span = (start, stop)
    return get_phrase(context, wordss, best_span)


def ensemble3(context, wordss, y1_list, y2_list):
    d = defaultdict(float)
    for y1, y2 in zip(y1_list, y2_list):
        span, score = get_best_span(y1, y2)
        phrase = get_phrase(context, wordss, span)
        d[phrase] += score
    return max(d.items(), key=lambda pair: pair[1])[0]


def ensemble4(context, wordss, y1_list, y2_list):
    d = defaultdict(lambda: 0.0)
    for y1, y2 in zip(y1_list, y2_list):
        for span, score in get_span_score_pairs(y1, y2):
            d[span] += score
    span = max(d.items(), key=lambda pair: pair[1])[0]
    phrase = get_phrase(context, wordss, span)
    return phrase


def combine_y_list(y_list, op='*'):
    if op == '+':
        func = sum
    elif op == '*':
        def func(l): return functools.reduce(mul, l)
    else:
        func = op
    return [[func(yij_list) for yij_list in zip(*yi_list)] for yi_list in zip(*y_list)]


def main():
    args = get_args()
    ensemble(args)

if __name__ == "__main__":
    main()


