import argparse
import csv
import json
import os
import random

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")

    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("--num", "-n", type=int, default=1000)
    parser.add_argument("--size", "-s", type=int, default=14000000)

    return parser.parse_args()


def sample_levy(args):
    in_path = args.in_path
    out_path = args.out_path
    in_num = args.size
    out_num = args.num
    i_set = set(random.sample(range(in_num), out_num))

    with open(in_path, 'r') as fin:
        with open(out_path, 'w') as fout:
            for i, line in tqdm(enumerate(fin), total=in_num):
                if i in i_set:
                    fout.write(line)

def main():
    args = get_args()
    sample_levy(args)

if __name__ == "__main__":
    main()
