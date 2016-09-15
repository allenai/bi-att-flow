from collections import deque
import json

import nltk
import re
import numpy as np

from tqdm import tqdm


def mytqdm(list_, desc="", show=True):
    if show:
        pbar = tqdm(list_)
        pbar.set_description(desc)
        return pbar
    return list_


def json_pretty_dump(obj, fh):
    return json.dump(obj, fh, sort_keys=True, indent=2, separators=(',', ': '))


def index(l, i):
    return index(l[i[0]], i[1:]) if len(i) > 1 else l[i[0]]


def fill(l, shape, dtype=None):
    out = np.zeros(shape, dtype=dtype)
    stack = deque()
    stack.appendleft(((), l))
    while len(stack) > 0:
        indices, cur = stack.pop()
        if len(indices) < shape:
            for i, sub in enumerate(cur):
                stack.appendleft([indices + (i,), sub])
        else:
            out[indices] = cur
    return out


def short_floats(o, precision):
    class ShortFloat(float):
        def __repr__(self):
            return '%.{}g'.format(precision) % self

    def _short_floats(obj):
        if isinstance(obj, float):
            return ShortFloat(obj)
        elif isinstance(obj, dict):
            return dict((k, _short_floats(v)) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return tuple(map(_short_floats, obj))
        return obj

    return _short_floats(o)


def argmax(x):
    return np.unravel_index(x.argmax(), x.shape)


def get_spans(text, tokens):
    """

    :param text:
    :param tokens:
    :return: the start indices of tokens in text
    """
    spans = []
    cur_idx = 0
    for token in tokens:
        if text.find(token, cur_idx) < 0:
            print(tokens)
            print("{} {} {}".format(token, cur_idx, text))
            raise Exception()
        cur_idx = text.find(token, cur_idx)
        spans.append((cur_idx, cur_idx+len(token)))
        cur_idx += len(token)
    return spans


def _find(token_spans, target_span):
    """

    :param token_spans: [ (0, 5), (6, 7), (9, 14), ... ]
    :param target_span: [ (5, 13) ]
    :return: (1, 3)
    """
    idxs = []
    for i, span in enumerate(token_spans):
        if not (target_span[1] <= span[0] or target_span[0] >= span[1]):
            idxs.append(i)
    return idxs[0], idxs[-1] + 1

def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            if text.find(token, cur_idx) < 0:
                print(tokens)
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss


def get_word_span(context, wordss, start, stop):
    spanss = get_2d_spans(context, wordss)
    idxs = []
    for sent_idx, spans in enumerate(spanss):
        for word_idx, span in enumerate(spans):
            if not (stop <= span[0] or start >= span[1]):
                idxs.append((sent_idx, word_idx))
    assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
    return idxs[0], (idxs[-1][0], idxs[-1][1] + 1)


def word_tokenize(text):
    temp_tokens = nltk.word_tokenize(text)
    tokens = []
    for token in temp_tokens:
        if token in ("``", "''"):
            tokens.append('"')
        else:
            flag = False
            for char in ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", ":", "\u2044",
                         "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0"):
                if char in token:
                    tokens.extend(re.split("({})".format(char), token))
                    flag = True
                    break
            if not flag:
                tokens.append(token)
    return tokens
