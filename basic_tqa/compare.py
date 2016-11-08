import sys
import json

first_path = sys.argv[1]
second_path = sys.argv[2]

a = json.load(open(first_path, 'r'))
b = json.load(open(second_path, 'r'))

assert len(a) == len(b)

diff_count = 0

for key, val in a.items():
    b_val = b[key]
    if val != b_val:
        print(val, "|||", b_val)
        diff_count += 1

print("{}/{} = {}".format(diff_count, len(a), diff_count/len(a)))