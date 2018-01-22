import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
import random
from collections import Counter

from tqdm import tqdm

from squad.utils import get_word_span, get_word_idx, process_tokens


def main():
    args = get_args()
    neg_squad(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    parser.add_argument("source_path")
    parser.add_argument("target_path")
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument('-r', "--aug_ratio", default=1, type=int)
    # TODO : put more args here
    return parser.parse_args()


def neg_squad(args):
    with open(args.source_path, 'r') as fp:
        squad = json.load(fp)
    with open(args.source_path, 'r') as fp:
        ref_squad = json.load(fp)

    for ai, article in enumerate(ref_squad['data']):
        for pi, para in enumerate(article['paragraphs']):
            cands = list(range(pi)) + list(range(pi+1, len(article['paragraphs'])))
            samples = random.sample(cands, args.aug_ratio)
            for sample in samples:
                for qi, ques in enumerate(article['paragraphs'][sample]['qas']):
                    new_ques = {'question': ques['question'], 'answers': [], 'answer_start': 0, 'id': "neg_" + ques['id']}
                    squad['data'][ai]['paragraphs'][pi]['qas'].append(new_ques)

    with open(args.target_path, 'w') as fp:
        json.dump(squad, fp)

if __name__ == "__main__":
    main()