import argparse
import json
import logging
import os

import itertools
import numpy as np
import nltk
from collections import OrderedDict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

import re

UNK = "<UNK>"
NULL = "<NULL>"

def _bool(str):
    if str == 'True':
        return True
    elif str == 'False':
        return False
    else:
        raise ValueError()


def _get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/model/squad"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=target_dir)
    # TODO : put more args here
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_word_size", default=100)
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--count_th", default=100, type=int)
    parser.add_argument("--para_size_th", default=8, type=int)
    parser.add_argument("--sent_size_th", default=64, type=int)
    parser.add_argument("--word_size_th", default=16, type=int)
    parser.add_argument("--char_count_th", default=500, type=int)
    parser.add_argument("--debug", default=False, type=_bool)
    return parser.parse_args()


def _prepro(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    glove_dir = args.glove_dir
    glove_corpus = args.glove_corpus
    glove_word_size = args.glove_word_size
    count_th = args.count_th
    para_size_th = args.para_size_th
    sent_size_th = args.sent_size_th
    word_size_th = args.word_size_th
    char_count_th = args.char_count_th
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
    train_shared = {'X': [], 'CX': []}  # X stores parass
    train_batched = {'*X': [], '*CX': [], 'Q': [], 'CQ': [], 'Y': [], 'ids': []}
    dev_shared = {'X': [], 'CX': []}  # X stores parass
    dev_batched = {'*X': [], '*CX': [], 'Q': [], 'CQ': [], 'Y': [], 'ids': []}
    params = {'emb_mat': []}

    train_path = os.path.join(source_dir, template.format("train", version))
    dev_path = os.path.join(source_dir, template.format("dev", version))

    _insert_raw_data(train_path, train_shared, train_batched, para_size_th=para_size_th, sent_size_th=sent_size_th)
    _insert_raw_data(dev_path, dev_shared, dev_batched, X_offset=len(train_shared['X']), para_size_th=para_size_th, sent_size_th=sent_size_th)

    word2vec_dict = _get_word2vec_dict(glove_path, train_shared, train_batched, total=total, count_th=count_th)
    word2idx_dict = {word: idx for idx, word in enumerate(word2vec_dict.keys())}  # Must be an ordered dict!
    char2idx_dict = _get_char2idx_dict(train_shared, train_batched, char_count_th=char_count_th)
    params['emb_mat'] = list(word2vec_dict.values())
    _apply(word2idx_dict, char2idx_dict, train_shared, train_batched, word_size_th=word_size_th)
    _apply(word2idx_dict, char2idx_dict, dev_shared, dev_batched, word_size_th=word_size_th)
    shared, _ = _concat(train=train_shared, dev=dev_shared, order=('train', 'dev'))
    batched, mode2idxs_dict = _concat(train=train_batched, dev=dev_batched, order=('train', 'dev'))

    _save(target_dir, shared, batched, params, mode2idxs_dict, word2idx_dict, char2idx_dict)


def _get_char2idx_dict(train_shared, train_batched, char_count_th=1000):
    chars = [NULL, UNK]
    counter = Counter(list(char for paras in train_shared['X'] for sents in paras for sent in sents for word in sent for char in word) +
        list(char for ques in train_batched['Q'] for word in ques for char in word))
    filtered_counter = {word: count for word, count in counter.items() if count >= char_count_th}
    chars.extend(filtered_counter.keys())
    char2idx_dict = dict(map(reversed, enumerate(chars)))
    return char2idx_dict


def _concat(order=None, **dict_dict):
    if order is not None:
        dicts = [dict_dict[key] for key in order]
    else:
        dicts = list(dict_dict.values())
    data = {key: list(itertools.chain(*[dict_[key] for dict_ in dicts])) for key in dicts[0]}
    mode2idxs_dict = {}
    count = 0
    for mode, dict_ in dict_dict.items():
        num = len(list(dict_.values())[0])
        mode2idxs_dict[mode] = list(range(count, count+num))
        count += num
    return data, mode2idxs_dict


def _print_stats(train_path, dev_path, min_count):
    train_shared = {'X': [], 'CX': []}  # X stores parass
    train_batched = {'*X': [], '*CX': [], 'Q': [], 'CQ': [], 'Y': [], 'ids': []}
    dev_shared = {'X': [], 'CX': []}  # X stores parass
    dev_batched = {'*X': [], '*CX': [], 'Q': [], 'CQ': [], 'Y': [], 'ids': []}
    _insert_raw_data(train_path, train_shared, train_batched)
    _insert_raw_data(dev_path, dev_shared, dev_batched)

    for min_count in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        print("-" * 80)
        print(min_count)
        train_word_counter = Counter([word for paras in train_shared['X'] for sents in paras for sent in sents for word in sent] +
                           [word for ques in train_batched['Q'] for word in ques])
        filtered_train_word_counter = {word: count for word, count in train_word_counter.items() if count > min_count}
        dev_word_counter = Counter([word for paras in dev_shared['X'] for sents in paras for sent in sents for word in sent] +
                           [word for ques in dev_batched['Q'] for word in ques])
        print("train words: {}, dev words: {}".format(sum(train_word_counter.values()), sum(dev_word_counter.values())))
        print("filtered train words: {}".format(sum(filtered_train_word_counter.values())))
        print("train words not observed filtered train: {}".format(sum(train_word_counter[word] for word in train_word_counter if word not in filtered_train_word_counter)))
        print("dev words not observed in filterd train: {}".format(sum(dev_word_counter[word] for word in dev_word_counter if word not in filtered_train_word_counter)))


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


def _insert_raw_data(file_path, raw_shared, raw_batched, X_offset=0, para_size_th=8, sent_size_th=32):
    START = "sstartt"
    STOP = "sstopp"
    X = raw_shared['X']
    RX, Q, Y, ids = raw_batched['*X'], raw_batched['Q'], raw_batched['Y'], raw_batched['ids']
    raw_batched['*CX'] = RX
    batched_idx = len(ids)  # = len(R) = len(Q) = len(Y)

    logging.info("reading {} ...".format(file_path))
    skip_count = 0
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
                if len(sents) > para_size_th:
                    logging.debug("Skipping para with num sents = {}".format(len(sents)))
                    skip_count += len(para['qas'])
                    continue
                max_sent_size = max(len(sent) for sent in sents)
                if max_sent_size > sent_size_th:
                    logging.debug("Skipping para with sent size = {}".format(max_sent_size))
                    skip_count += len(para['qas'])
                    continue

                X_i.append(sents)
                assert context.find(START) < 0 and context.find(STOP) < 0, "Choose other start, stop words"
                for qa in para['qas']:
                    id_ = qa['id']
                    question = qa['question']
                    question_words = _tokenize(question)[0]
                    if len(question_words) > sent_size_th:
                        logging.debug("Skipping ques with sent size = {}".format(len(question_words)))
                        skip_count += 1
                        continue

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
                        if len(sents) <= start_idx[0] or len(sents[start_idx[0]]) <= start_idx[1]:
                            logging.warning("Skipping qa id {}: invalid answer annotation".format(id_))
                            continue
                        temp_idx = _index(temp_sents, STOP, 2)
                        stop_idx = temp_idx[0], temp_idx[1] - 1

                        # Store stuff
                        RX.append(ref_idx)
                        Q.append(question_words)
                        Y.append(start_idx)
                        ids.append(id_)
                        batched_idx += 1
                        continue  # considering only one answer for now
                # break  # for debugging
        if counter > 0:
            logging.warning("# answer mismatches: {}".format(counter))
        logging.info("# skipped questions: {}".format(skip_count))
        logging.info("# articles: {}, # paragraphs: {}".format(len(X), sum(len(x) for x in X)))
        logging.info("# questions: {}".format(len(Q)))

        # Stats
        """
        X_num_words_counter = Counter(len(sent) for paras in X for sents in paras for sent in sents)
        X_num_sents_counter = Counter(len(sents) for paras in X for sents in paras)
        Q_num_words_counter = Counter(len(ques) for ques in Q)
        X_num_chars_counter = Counter(len(word) for paras in X for sents in paras for sent in sents for word in sent)
        Q_num_chars_counter = Counter(len(word) for ques in Q for word in ques)
        plt.plot(list(X_num_words_counter.keys()), list(X_num_words_counter.values()))
        plt.show()
        plt.plot(list(X_num_sents_counter.keys()), list(X_num_sents_counter.values()))
        plt.show()
        plt.plot(list(Q_num_words_counter.keys()), list(Q_num_words_counter.values()))
        plt.show()
        plt.plot(list(X_num_chars_counter.keys()), list(X_num_chars_counter.values()))
        plt.show()
        plt.plot(list(Q_num_chars_counter.keys()), list(Q_num_chars_counter.values()))
        plt.show()
        """


def _get_word2vec_dict(glove_path, shared, batched, total=None, count_th=0):
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
    # count filtering
    word2vec_dict = {word: vec for word, vec in word2vec_dict.items() if word_counter[word] > count_th}

    unk_word_counter = {word: count for word, count in word_counter.items() if word not in word2vec_dict}
    top_unk_words = [word for word, _ in sorted(unk_word_counter.items(), key=lambda pair: -pair[1])][:10]
    longest_words = list(sorted(word_counter.keys(), key=lambda x: len(x)))[-10:]
    total_count = sum(word_counter.values())
    unk_count = sum(unk_word_counter.values())
    logging.info("# known words: {}, # unk words: {}".format(total_count, unk_count))
    logging.info("# distinct known words: {}, # distinct unk words: {}".format(len(word2vec_dict), len(word_counter)-len(word2vec_dict)))
    logging.info("Top unk words: {}".format(", ".join(top_unk_words)))
    logging.info("Longest words: {}".format(", ".join(longest_words)))
    word2vec_dict[UNK] = [0.0] * word_vec_size
    return word2vec_dict


def _apply(word2idx_dict, char2idx_dict, shared, batched, word_size_th=16):
    def _get(word):
        if word not in word2idx_dict:
            word = UNK
        return word2idx_dict[word]

    def _get_chars(word):
        if len(word) > word_size_th:
            word = word[:word_size_th]
        return [(char2idx_dict[char] if char in char2idx_dict else char2idx_dict[UNK]) for char in word]

    logging.info("applying word2idx_dict to data ...")
    X = [[[[_get(word) for word in sent] for sent in sents] for sents in paras]for paras in tqdm(shared['X'])]
    Q = [[_get(word) for word in ques] for ques in tqdm(batched['Q'])]
    CX = [[[[_get_chars(word) for word in sent] for sent in sents] for sents in paras]for paras in tqdm(shared['X'])]
    CQ = [[_get_chars(word) for word in ques] for ques in tqdm(batched['Q'])]
    shared['X'] = X
    batched['Q'] = Q
    shared['CX'] = CX
    batched['CQ'] = CQ


def _save(target_dir, shared, batched, params, mode2idxs_dict, word2idx_dict, char2idx_dict):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    mode2idxs_path = os.path.join(target_dir, "mode2idxs.json")
    metadata_path = os.path.join(target_dir, "metadata.json")
    shared_path = os.path.join(target_dir, "shared.json")
    batched_path =os.path.join(target_dir, "batched.json")
    word2idx_path = os.path.join(target_dir, "word2idx.json")
    param_path = os.path.join(target_dir, "param.json")
    char2idx_path = os.path.join(target_dir, "char2idx.json")

    CX, X = shared['CX'], shared['X']
    emb_mat = params['emb_mat']
    RX, RCX, Q, CQ, Y = (batched[key] for key in ('*X', '*CX', 'Q', 'CQ', 'Y'))

    max_word_size = max(max(len(word) for paras in CX for sents in paras for sent in sents for word in sent),
                        max(len(word) for ques in CQ for word in ques))

    metadata = {'max_sent_size': max(len(sent) for paras in X for sents in paras for sent in sents),
                'max_num_sents': max(len(sents) for paras in X for sents in paras),
                'vocab_size': len(emb_mat),
                'char_vocab_size': len(char2idx_dict),
                'max_ques_size': max(len(ques) for ques in Q),
                'max_word_size': max_word_size,
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
    with open(char2idx_path, 'w') as fh:
        json.dump(char2idx_dict, fh)


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = _get_args()
    _prepro(args)


if __name__ == "__main__":
    main()
