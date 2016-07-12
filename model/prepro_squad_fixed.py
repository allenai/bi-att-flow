import argparse
import json
import logging
import os

import itertools
import numpy as np
import nltk
from collections import OrderedDict, Counter
from tqdm import tqdm

import re

UNK = "<UNK>"


def _get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/model/squad"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=target_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_word_size", default=100)
    parser.add_argument("--version", default="1.0")
    # TODO : put more args here
    return parser.parse_args()


def _prepro(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    glove_dir = args.glove_dir
    glove_corpus = args.glove_corpus
    glove_word_size = args.glove_word_size
    # TODO : formally specify this path
    glove_path = os.path.join(glove_dir, "glove.{}.{}d.txt".format(glove_corpus, glove_word_size))
    if glove_corpus == '6B':
        total = int(4e5)
    elif glove_corpus == '42B':
        total = int(1.9e6)
    elif glove_corpus == '840B':
        total = int(2.2e6)
    elif glove_corpus == '2B':
        total = int(1.2e6)
    else:
        raise ValueError()
    # TODO : put something here; Fake data shown
    version = args.version
    template = "{}-v{}.json"
    shared = {'X': []}  # X stores parass
    batched = {'R': [], 'Q': [], 'Y': [], 'ids': []}
    params = {'emb_mat': []}
    mode2idxs_dict = {'train': [], 'dev': []}


    train_path = os.path.join(source_dir, template.format("train", version))
    dev_path = os.path.join(source_dir, template.format("dev", version))

    _insert_raw_data(train_path, shared, batched, mode2idxs_dict, 'train')
    _insert_raw_data(dev_path, shared, batched, mode2idxs_dict, 'dev')


    word2vec_dict = _get_word2vec_dict(glove_path, shared, batched, total=total)
    word2idx_dict = {word: idx for idx, word in enumerate(word2vec_dict.keys())}  # Must be an ordered dict!
    params['emb_mat'] = list(word2vec_dict.values())
    _apply(word2idx_dict, shared, batched)
    _save(target_dir, shared, batched, params, mode2idxs_dict, word2idx_dict)


def _print_stats(train_path, dev_path):
    train_shared = {'X': []}  # X stores parass
    train_batched = {'R': [], 'Q': [], 'Y': [], 'ids': []}
    dev_shared = {'X': []}  # X stores parass
    dev_batched = {'R': [], 'Q': [], 'Y': [], 'ids': []}
    mode2idxs_dict = {'train': [], 'dev': []}
    _insert_raw_data(train_path, train_shared, train_batched, mode2idxs_dict, 'train')
    _insert_raw_data(dev_path, dev_shared, dev_batched, mode2idxs_dict, 'dev')
    train_word_counter = Counter([word for paras in train_shared['X'] for sents in paras for sent in sents for word in sent] +
                           [word for ques in train_batched['Q'] for word in ques])
    dev_word_counter = Counter([word for paras in dev_shared['X'] for sents in paras for sent in sents for word in sent] +
                           [word for ques in dev_batched['Q'] for word in ques])
    print("train words: {}, dev words: {}".format(sum(train_word_counter.values()), sum(dev_word_counter.values())))
    print("dev words not observed in train: {}".format(sum(dev_word_counter[word] for word in dev_word_counter if word not in train_word_counter)))


def _tokenize(raw):
    raw = raw.lower()
    sents = nltk.sent_tokenize(raw)
    tokens = [nltk.word_tokenize(sent) for sent in sents]
    return tokens


def _index(l, w, d):
    if d == 1:
        return [l.index(w)]
    for i, ll in enumerate(l):
        try:
            return [i] + _index(ll, w, d-1)
        except ValueError:
            continue
    raise ValueError("{} is not in list".format(w))


def _insert_raw_data(file_path, raw_shared, raw_batched, mode2idxs_dict, mode):
    START = "sstartt"
    STOP = "sstopp"
    X = raw_shared['X']
    R, Q, Y, ids = raw_batched['R'], raw_batched['Q'], raw_batched['Y'], raw_batched['ids']
    idxs = mode2idxs_dict[mode]
    X_offset = len(X)
    batched_idx = len(ids)  # = len(R) = len(Q) = len(Y)

    logging.info("reading {} ...".format(file_path))
    with open(file_path, 'r') as fh:
        d = json.load(fh)
        counter = 0
        for article_idx, article in enumerate(tqdm(d['data'])):
            X_i = []
            X.append(X_i)
            for para in article['paragraphs']:
                para_idx = len(X_i)
                ref_idx = (article_idx + X_offset, para_idx)
                context = para['context']
                sents = _tokenize(context)
                max_sent_size = max(len(sent) for sent in sents)
                if max_sent_size > 200:
                    logging.warning("Skipping para with sent size = {}".format(max_sent_size))
                    continue
                X_i.append(sents)
                assert context.find(START) < 0 and context.find(STOP) < 0, "Choose other start, stop words"
                for qa in para['qas']:
                    id_ = qa['id']
                    question = qa['question']
                    question_words = _tokenize(question)[0]
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_words = _tokenize(text)
                        answer_start = answer['answer_start']
                        answer_stop = answer_start + len(text)
                        candidate_words = _tokenize(context[answer_start:answer_stop])
                        if answer_words != candidate_words:
                            logging.debug("Mismatching answer found: '{}', '{}'".format(candidate_words, answer_words))
                            counter += 1
                        temp_context = "{} {} {} {} {}".format(context[:answer_start], START,
                                                               context[answer_start:answer_stop], STOP,
                                                               context[answer_stop:])
                        temp_sents = _tokenize(temp_context)
                        start_idx = _index(temp_sents, START, 2)
                        temp_idx = _index(temp_sents, STOP, 2)
                        stop_idx = temp_idx[0], temp_idx[1] - 1

                        # Store stuff
                        R.append(ref_idx)
                        Q.append(question_words)
                        Y.append(start_idx)
                        ids.append(id_)
                        idxs.append(batched_idx)
                        batched_idx += 1
                        continue  # considering only one answer for now
            # break  # for debugging
        if counter > 0:
            logging.warning("# answer mismatches: {}".format(counter))
        logging.info("# articles: {}, # paragraphs: {}".format(len(X), sum(len(x) for x in X)))
        logging.info("# questions: {}".format(len(Q)))


def _get_word2vec_dict(glove_path, shared, batched, total=None):
    word_counter = Counter([word for paras in shared['X'] for sents in paras for sent in sents for word in sent] +
                           [word for ques in batched['Q'] for word in ques])
    word2vec_dict = OrderedDict()

    logging.info("reading %s ... " % glove_path)
    word_vec_size = None
    with open(glove_path, 'r') as fp:
        for line in tqdm(fp, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            if word in word_counter:
                vector = list(map(float, array[1:]))
                word_vec_size = len(vector)
                word2vec_dict[word] = vector
    unk_word_counter = {word: count for word, count in word_counter.items() if word not in word2vec_dict}
    top_unk_words = [word for word, _ in sorted(unk_word_counter.items(), key=lambda pair: -pair[1])][:10]
    total_count = sum(word_counter.values())
    unk_count = sum(unk_word_counter.values())
    logging.info("# known words: {}, # unk words: {}".format(total_count, unk_count))
    logging.info("# distinct known words: {}, # distinct unk words: {}".format(len(word2vec_dict), len(word_counter)-len(word2vec_dict)))
    logging.info("Top unk words: {}".format(", ".join(top_unk_words)))
    word2vec_dict[UNK] = [0.0] * word_vec_size
    return word2vec_dict


def _apply(word2idx_dict, shared, batched):
    def _get(word):
        if word not in word2idx_dict:
            word = UNK
        return word2idx_dict[word]
    logging.info("applying word2idx_dict to data ...")
    X = [[[[_get(word) for word in sent] for sent in sents] for sents in paras]for paras in tqdm(shared['X'])]
    Q = [[_get(word) for word in ques] for ques in tqdm(batched['Q'])]
    shared['X'] = X
    batched['Q'] = Q


def _save(target_dir, shared, batched, params, mode2idxs_dict, word2idx_dict):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    mode2idxs_path = os.path.join(target_dir, "mode2idxs.json")
    metadata_path = os.path.join(target_dir, "metadata.json")
    shared_path = os.path.join(target_dir, "shared.json")
    batched_path =os.path.join(target_dir, "batched.json")
    word2idx_path = os.path.join(target_dir, "word2idx.json")
    param_path = os.path.join(target_dir, "param.json")

    X = shared['X']
    emb_mat = params['emb_mat']
    R, Q, Y = (batched[key] for key in ('R', 'Q', 'Y'))

    metadata = {'max_sent_size': max(len(sent) for paras in X for sents in paras for sent in sents),
                'max_num_sents': max(len(sents) for paras in X for sents in paras),
                'vocab_size': len(emb_mat),
                'max_ques_size': max(len(ques) for ques in Q),
                "word_vec_size": len(emb_mat[0]),
                }

    logging.info("saving ...")
    with open(mode2idxs_path, 'w') as fh:
        json.dump(mode2idxs_dict, fh)
    with open(metadata_path, 'w') as fh:
        json.dump(metadata, fh)
    with open(shared_path, 'w') as fh:
        json.dump(shared, fh)
    with open(batched_path, 'w') as fh:
        json.dump(batched, fh)
    with open(word2idx_path, 'w') as fh:
        json.dump(word2idx_dict, fh)
    with open(param_path, 'w') as fh:
        json.dump(params, fh)


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = _get_args()
    _prepro(args)


if __name__ == "__main__":
    main()
