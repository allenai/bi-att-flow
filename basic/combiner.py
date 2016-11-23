import sys
import json
from collections import Counter
import re

def key_func(pair):
    return pair[1]


def get_func(vals, probs):
    counter = Counter(vals)
    return max(zip(vals, probs), key=lambda pair: pair[1] + 0.5 * counter[pair[0]] / len(counter) - 999 * (len(pair[0]) == 0) )[0]

def normalize(val):
    val = val.replace(" 's", "'s")
    return val

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
    vals = [normalize(val) for val in vals]
    largest_val = get_func(vals, probs)
    c[key] = largest_val

json.dump(c, open(third_path, 'w'))


