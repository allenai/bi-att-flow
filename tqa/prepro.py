import argparse
import json
import os

from collections import Counter

from tqdm import tqdm

import nltk


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
    source_path = os.path.join(home, "data", "tqa", "beta5p5.json")
    target_dir = "data/tqa"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument("--source_path", default=source_path)
    parser.add_argument("--target_dir", default=target_dir)
    parser.add_argument("--debug", default=False, type=bool_)
    parser.add_argument("--train_ratio", default=0.7, type=float)
    parser.add_argument("--dev_ratio", default=0.1, type=float)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="split", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--process_tokens", default=False, type=bool_)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--merge", default=False, type=bool_)
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, "train-v1.1.json")
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, "dev-v1.1.json")
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.mode == 'full':
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')
        prepro_each(args, 'dev', out_name='test')
    elif args.mode =='split':
        prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
        prepro_each(args, 'train', args.train_ratio, args.train_ratio + args.dev_ratio, out_name='dev')
        prepro_each(args, 'dev', args.train_ratio + args.dev_ratio, 1.0, out_name='test')
    else:
        raise Exception()


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


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    sent_tokenize = nltk.sent_tokenize
    if args.merge:
        sent_tokenize = lambda para: [para]

    def word_tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

    source_data = json.load(open(args.source_path, 'r'))

    q, cq, y, rx, rcx, ids = [], [], [], [], [], []
    x, cx = [], []
    a, ca = [], []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()

    start_ci = int(round(len(source_data) * start_ratio))
    stop_ci = int(round(len(source_data) * stop_ratio))
    skip_count = 0
    for ci, chapter in enumerate(tqdm(source_data[start_ci:stop_ci])):
        xi = []
        cxi = []
        for tj, topic in chapter['topics'].items():
            cur = topic['content']['text']
            cur = cur.replace("''", '" ')
            cur = cur.replace("``", '" ')
            xij = list(map(word_tokenize, sent_tokenize(cur)))
            cxij = [[list(xijkl) for xijkl in xijk] for xijk in xij]
            xi.append(xij)
            cxi.append(cxij)

            for xijk in xij:
                for xijkl in xijk:
                    l = len(chapter['questions']['nonDiagramQuestions'])
                    word_counter[xijkl] += l
                    lower_word_counter[xijkl.lower()] += l
                    for xijklm in xijkl:
                        char_counter[xijklm] += l

        x.append(xi)
        cx.append(cxi)

        if len(xi) == 0 or sum(map(len, xi)) == 0:
            skip_count += len(chapter['questions']['nonDiagramQuestions'])
            continue


        for qid, question in chapter['questions']['nonDiagramQuestions'].items():
            if 'processedText' not in question['correctAnswer']:
                # print("Skipping question '{}' because no processed text ...".format(qid))
                skip_count += 1
                continue
            if len(question['answerChoices']) == 0:
                # print("Skipping question '{}' because no answer choices ...".format(qid))
                skip_count += 1
                continue
            qi = word_tokenize(question['beingAsked']['processedText'])
            for qij in qi:
                word_counter[qij] += 1
                lower_word_counter[qij.lower()] += 1
                for qijk in qij:
                    char_counter[qijk] += 1

            cqi = list(map(list, qi))
            yi_raw = question['correctAnswer']['processedText']
            yi = None
            ai = []
            cai = []
            for aid, answer in question['answerChoices'].items():
                if aid == yi_raw or (yi_raw in ('true', 'false') and yi_raw == answer['processedText']):
                    yi = len(ai)
                aij = word_tokenize(answer['processedText'])
                ai.append(aij)
                caij = list(map(list, aij))
                cai.append(caij)
                for aijk in aij:
                    word_counter[aijk] += 1
                    lower_word_counter[aijk.lower()] += 1
                    for aijkl in aijk:
                        char_counter[aijkl] += 1

            if yi is None:
                # print("Skipping question '{}' because answer does not match the choices ...".format(qid))
                skip_count += 1
                continue

            q.append(qi)
            cq.append(cqi)
            y.append(yi)
            a.append(ai)
            ca.append(cai)
            rx.append(ci)
            rcx.append(ci)
            ids.append(question['globalID'])
        if args.debug:
            break

    print("n={}, skipped={}".format(len(q), skip_count))

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'ids': ids, 'a': a, 'ca': ca}
    shared = {'x': x, 'cx': cx,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)


if __name__ == "__main__":
    main()