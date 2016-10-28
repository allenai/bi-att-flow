import sys
import json

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
    shortest_val = max(zip(vals, probs), key=lambda pair: pair[1])[0]
    c[key] = shortest_val

json.dump(c, open(third_path, 'w'))
