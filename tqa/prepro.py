import argparse
import json
import os

from collections import Counter

import itertools
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
    source_path = os.path.join(home, "data", "tqa", "beta7.5.json")
    caption_dir = os.path.join(home, "data", "tqa", "captions")
    split_path =os.path.join(home, "data", "tqa", "tt_split.json")
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
    parser.add_argument("--qtype", default='both', type=str)
    parser.add_argument("--caption_dir", default=caption_dir, type=str)
    parser.add_argument("--split_path", default=split_path, type=str)
    parser.add_argument("--mc_only", default=True, type=bool_)
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
        with open(args.split_path, 'r') as fh:
            split = json.load(fh)
            train_ids = split['train']
            test_ids = split['test']
        prepro_each(args, train_ids, out_name='train')
        prepro_each(args, test_ids, out_name='dev')
        prepro_each(args, test_ids, out_name='test')
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


def get_diagram(args, image_name):
    path = os.path.join(args.caption_dir, "{}.json".format(image_name))
    with open(path, 'r') as fh:
        dt = json.load(fh)
        return dt


def prepro_each(args, lesson_ids, out_name="default", in_path=None):
    sent_tokenize = nltk.sent_tokenize
    if args.merge:
        sent_tokenize = lambda para: [para]

    def word_tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

    source_data = json.load(open(args.source_path, 'r'))

    q, cq, y, rx, rcx, ids = [], [], [], [], [], []
    x, cx = [], []
    a, ca = [], []
    d, cd = [], []
    t = []

    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()

    skip_count = 0
    lesson_ids = set(lesson_ids)
    chapters = [chapter for chapter in source_data if chapter['globalID'] in lesson_ids]
    for ci, chapter in enumerate(tqdm(chapters)):
        if args.qtype == 'diagram':
            questions = chapter['questions']['diagramQuestions']
        elif args.qtype == 'text':
            questions = chapter['questions']['nonDiagramQuestions']
        elif args.qtype == 'both':
            questions = dict(itertools.chain(chapter['questions']['diagramQuestions'].items(), chapter['questions']['nonDiagramQuestions'].items()))
        else:
            raise Exception()
        if args.mc_only:
            questions = {qid: question for qid, question in questions.items() if question['questionType'] in ['Multiple Choice', 'Diagram Multiple Choice']}

        xi = []
        cxi = []
        paras = []
        # text
        for tj, topic in chapter['topics'].items():
            text = topic['content']['text']
            paras.append(sent_tokenize(text))
            for figure in topic['content']['figures']:
                caption = sent_tokenize(figure['caption'])
                paras.append(caption)

        # inst diagrams
        for _, inst in chapter['instructionalDiagrams'].items():
            image_name = inst['imageName']
            diagram = get_diagram(args, image_name)
            inst_text = sent_tokenize(inst['processedText'])
            paras.append(diagram)
            paras.append(inst_text)

        for para in paras:
            para = [sent.replace("''", '" ').replace("``", '" ') for sent in para]
            xij = list(map(word_tokenize, para))
            cxij = [[list(xijkl) for xijkl in xijk] for xijk in xij]
            xi.append(xij)
            cxi.append(cxij)

            for xijk in xij:
                for xijkl in xijk:
                    l = len(questions)
                    word_counter[xijkl] += l
                    lower_word_counter[xijkl.lower()] += l
                    for xijklm in xijkl:
                        char_counter[xijklm] += l

        x.append(xi)
        cx.append(cxi)

        if len(xi) == 0 or sum(map(len, xi)) == 0:
            skip_count += len(questions)
            continue

        for qid, question in questions.items():
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

            di = []
            cdi = []
            # diagram descriptions
            if 'imageName' in question:
                image_name = question['imageName']
                dt = get_diagram(args, image_name)
                di = list(map(word_tokenize, dt))
                cdi = [[list(word) for word in sent] for sent in di]
                for dij in di:
                    for dijk in dij:
                        word_counter[dijk] += 1
                for cdij in cdi:
                    for cdijk in cdij:
                        for cdijkl in cdijk:
                            char_counter[cdijkl] += 1

            # choices and answers
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
            d.append(di)
            cd.append(cdi)
            t.append(question['questionType'])
        if args.debug:
            break

    print("n={}, skipped={}".format(len(q), skip_count))

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'ids': ids, 'a': a, 'ca': ca, 'd': d, 'cd': cd}
    shared = {'x': x, 'cx': cx,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)


if __name__ == "__main__":
    main()