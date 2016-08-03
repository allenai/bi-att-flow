import argparse
import json
import os

from tqdm import tqdm

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

def _augment(ih, dict_, key, is_doc):
    assert isinstance(ih, CoreNLPInterface)
    content = dict_[key]
    if is_doc:
        sents = ih.split_doc(content)
    else:
        sents = [content]
    words = list(map(ih.split_sent, sents))
    # const = list(map(ih.get_const_str, sents))
    dep = list(map(ih.get_dep, sents))
    if not is_doc:
        words = words[0]
        # const = const[0]
        dep = dep[0]
    dict_["{}_words".format(key)] = words
    # dict_["{}_const".format(key)] = const
    dict_["{}_dep".format(key)] = dep


def _prepro(args):
    in_path = args.in_path
    out_path = args.out_path
    ih = CoreNLPInterface(args.domain, args.port)

    with open(in_path, 'r') as fh:
        d = json.load(fh)
        size = sum(len(article['paragraphs']) for article in d['data'])
        pbar = tqdm(range(size))
        for article in d['data']:
            for para in article['paragraphs']:
                pbar.update(1)
                context = para['context']
                _augment(ih, para, 'context', True)
                for qa in para['qas']:
                    question = qa['question']
                    _augment(ih, qa, 'question', False)
                    for answer in qa['answers']:
                        text = answer['text']
                        _augment(ih, answer, 'text', False)
        pbar.close()

    with open(out_path, 'w') as fh:
        json.dump(d, fh)


def _main():
    args = _get_args()
    _prepro(args)


if __name__ == "__main__":
    _main()