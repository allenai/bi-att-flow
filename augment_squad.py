import argparse
import json
import os

from tqdm import tqdm
import nltk

from corenlp_interface import CoreNLPInterface


def _get_args():
    parser = argparse.ArgumentParser()
    # home = os.path.expanduser("~")
    domain = "127.0.0.1"  # "vision-server2.corp.ai2"
    port = 8000
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("--domain", default=domain)
    parser.add_argument("--port", default=port)
    return parser.parse_args()


def _index(l, w, d):
    if d == 1:
        return [l.index(w)]
    for i, ll in enumerate(l):
        try:
            return [i] + _index(ll, w, d-1)
        except ValueError:
            continue
    raise ValueError("{} is not in list".format(w))


def _augment(ih, dict_, key, is_doc):
    assert isinstance(ih, CoreNLPInterface)
    content = dict_[key]
    if is_doc:
        sents = nltk.sent_tokenize(content)
    else:
        sents = [content]
    # words = list(map(ih.split_sent, sents))
    const = list(map(ih.get_const, sents))
    dep = list(map(ih.get_dep, sents))
    if not is_doc:
        const = const[0]
        dep = dep[0]
    dict_["{}_const".format(key)] = const
    dict_["{}_dep".format(key)] = dep
    if is_doc:
        return sum(each is None for each in dep)
    return int(dep is None)


def _prepro(args):
    in_path = args.in_path
    out_path = args.out_path
    ih = CoreNLPInterface(args.domain, args.port)
    counter = 0
    mismatch_counter = 0

    START = "SSSTARTTT"
    STOP = "SSSTOPPP"

    with open(in_path, 'r') as fh:
        d = json.load(fh)
        size = sum(len(article['paragraphs']) for article in d['data'])
        pbar = tqdm(range(size))
        for article in d['data']:
            for para in article['paragraphs']:
                pbar.update(1)
                context = para['context']
                counter += _augment(ih, para, 'context', True)
                for i, each in enumerate(para['context_dep']):
                    print("*{}*".format(i), each)
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
                for qa in para['qas']:
                    question = qa['question']
                    counter += _augment(ih, qa, 'question', False)
                    for answer in qa['answers']:
                        text = answer['text']
                        text_words = ih.split_sent(text)
                        answer_start = answer['answer_start']
                        dirty_context = "{} {} {}".format(context[:answer_start], START, context[answer_start:])
                        wordss = list(map(ih.split_sent, nltk.sent_tokenize(dirty_context)))
                        start_idx = _index(wordss, START, 2)
                        stop_idx = [start_idx[0], start_idx[1] + len(text_words)]
                        answer['start_idx'] = start_idx
                        answer['stop_idx'] = stop_idx
                        answer_words = [each[0] for each in context_nodes[start_idx[0]][start_idx[1]:stop_idx[1]]]
                        if answer_words != text_words:
                            print(answer_words, text_words)
                            mismatch_counter += 1
                # break
            # print(counter)
            # print(mismatch_counter)
        print(counter, mismatch_counter)
        pbar.close()

    with open(out_path, 'w') as fh:
        json.dump(d, fh)


def _main():
    args = _get_args()
    _prepro(args)


if __name__ == "__main__":
    _main()