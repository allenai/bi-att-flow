import sys
import json
from collections import Counter, defaultdict
import re

def key_func(pair):
    return pair[1]


def get_func(vals, probs):
    counter = Counter(vals)
    # return max(zip(vals, probs), key=lambda pair: pair[1])[0]
    # return max(zip(vals, probs), key=lambda pair: pair[1] * counter[pair[0]] / len(counter) - 999 * (len(pair[0]) == 0) )[0]
    # return max(zip(vals, probs), key=lambda pair: pair[1] + 0.7 * counter[pair[0]] / len(counter) - 999 * (len(pair[0]) == 0) )[0]
    d = defaultdict(float)
    for val, prob in zip(vals, probs):
        d[val] += prob
    d[''] = 0
    return max(d.items(), key=lambda pair: pair[1])[0]

third_path = sys.argv[1]
other_paths = sys.argv[2:]

others = [json.load(open(path, 'r')) for path in other_paths]


c = {}

assert min(map(len, others)) == max(map(len, others)), list(map(len, others))

for key in others[0].keys():
    if key == 'scores':
        continue
    probs = [other['scores'][key] for other in others]
    vals = [other[key] for other in others]
    largest_val = get_func(vals, probs)
    c[key] = largest_val

json.dump(c, open(third_path, 'w'))