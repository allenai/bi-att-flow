import os
import sys
import json, pickle, gzip

data_path = "data/tqa/data_test.json"
eval_path = sys.argv[1]
out_path = sys.argv[2]

with open(data_path, 'r') as fp:
    data = json.load(fp)

with gzip.open(eval_path, 'r') as fp:
    e = pickle.load(fp)

out = {'ids': data['ids'], 'correct': e['correct'], 'yp': e['yp']}

with open(out_path, 'w') as fp:
    json.dump(out, fp)
