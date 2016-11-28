import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm

from my.utils import process_tokens
from squad.utils import get_word_span, process_tokens


def bool_(arg):
    if arg == 'True':
        return True
    elif arg == 'False':
        return False
    raise Exception(arg)


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "cnn", 'questions')
    target_dir = "data/cnn"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=target_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_corpus", default='6B')
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--debug", default=False, type=bool_)
    parser.add_argument("--num_sents_th", default=200, type=int)
    parser.add_argument("--ques_size_th", default=30, type=int)
    parser.add_argument("--width", default=5, type=int)
    # TODO : put more args here
    return parser.parse_args()


def prepro(args):
    prepro_each(args, 'train')
    prepro_each(args, 'dev')
    prepro_each(args, 'test')


def para2sents(para, width):
    """
    Turn para into double array of words (wordss)
    Where each sentence is up to 5 word neighbors of each entity
    :param para:
    :return:
    """
    words = para.split(" ")
    sents = []
    for i, word in enumerate(words):
        if word.startswith("@"):
            start = max(i - width, 0)
            stop = min(i + width + 1, len(words))
            sent = words[start:stop]
            sents.append(sent)
    return sents


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
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


def prepro_each(args, mode):
    source_dir = os.path.join(args.source_dir, mode)
    word_counter = Counter()
    lower_word_counter = Counter()
    ent_counter = Counter()
    char_counter = Counter()
    max_sent_size = 0
    max_word_size = 0
    max_ques_size = 0
    max_num_sents = 0

    file_names = list(os.listdir(source_dir))
    if args.debug:
        file_names = file_names[:1000]
    lens = []

    out_file_names = []
    for file_name in tqdm(file_names, total=len(file_names)):
        if file_name.endswith(".question"):
            with open(os.path.join(source_dir, file_name), 'r') as fh:
                url = fh.readline().strip()
                _ = fh.readline()
                para = fh.readline().strip()
                _ = fh.readline()
                ques = fh.readline().strip()
                _ = fh.readline()
                answer = fh.readline().strip()
                _ = fh.readline()
                cands = list(line.strip() for line in fh)
                cand_ents = list(cand.split(":")[0] for cand in cands)
                sents = para2sents(para, args.width)
                ques_words = ques.split(" ")

                # Filtering
                if len(sents) > args.num_sents_th or len(ques_words) > args.ques_size_th:
                    continue

                max_sent_size = max(max(map(len, sents)), max_sent_size)
                max_ques_size = max(len(ques_words), max_ques_size)
                max_word_size = max(max(len(word) for sent in sents for word in sent), max_word_size)
                max_num_sents = max(len(sents), max_num_sents)

                for word in ques_words:
                    if word.startswith("@"):
                        ent_counter[word] += 1
                        word_counter[word] += 1
                    else:
                        word_counter[word] += 1
                        lower_word_counter[word.lower()] += 1
                        for c in word:
                            char_counter[c] += 1
                for sent in sents:
                    for word in sent:
                        if word.startswith("@"):
                            ent_counter[word] += 1
                            word_counter[word] += 1
                        else:
                            word_counter[word] += 1
                            lower_word_counter[word.lower()] += 1
                            for c in word:
                                char_counter[c] += 1

                out_file_names.append(file_name)
                lens.append(len(sents))
    num_examples = len(out_file_names)

    assert len(out_file_names) == len(lens)
    sorted_file_names, lens = zip(*sorted(zip(out_file_names, lens), key=lambda each: each[1]))
    assert lens[-1] == max_num_sents

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dit = get_word2vec(args, lower_word_counter)

    shared = {'word_counter': word_counter, 'ent_counter': ent_counter, 'char_counter': char_counter,
              'lower_word_counter': lower_word_counter,
              'max_num_sents': max_num_sents, 'max_sent_size': max_sent_size, 'max_word_size': max_word_size,
              'max_ques_size': max_ques_size,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dit, 'sorted': sorted_file_names,
              'num_examples': num_examples}

    print("max num sents: {}".format(max_num_sents))
    print("max ques size: {}".format(max_ques_size))

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(mode))
    with open(shared_path, 'w') as fh:
        json.dump(shared, fh)


if __name__ == "__main__":
    main()
