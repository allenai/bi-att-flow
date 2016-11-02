import json
import os
import sys

root_dir = sys.argv[1]
answer_path = sys.argv[2]
file_names = os.listdir(root_dir)

num_correct = 0
num_wrong = 0

with open(answer_path, 'r') as fh:
    id2answer_dict = json.load(fh)

for file_name in file_names:
    if not file_name.endswith(".question"):
        continue
    with open(os.path.join(root_dir, file_name), 'r') as fh:
        url = fh.readline().strip()
        _ = fh.readline()
        para = fh.readline().strip()
        _ = fh.readline()
        ques = fh.readline().strip()
        _ = fh.readline()
        answer = fh.readline().strip()
        _ = fh.readline()
        if file_name in id2answer_dict:
            pred = id2answer_dict[file_name]
            if pred == answer:
                num_correct += 1
            else:
                num_wrong += 1
        else:
            num_wrong += 1

total = num_correct + num_wrong
acc = float(num_correct) / total
print("{} = {} / {}".format(acc, num_correct, total))