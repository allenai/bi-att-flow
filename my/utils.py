from collections import deque
import json

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


