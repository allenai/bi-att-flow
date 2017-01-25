import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-t", "--th", type=float, default=0.5)
    # TODO : put more args here
    return parser.parse_args()


def get_pr(args):
    with open(args.path, 'r') as fp:
        answers = json.load(fp)

    na = answers['na']

    tp = sum(int(not id_.startswith("neg") and score < args.th) for id_, score in na.items())
    fp = sum(int(id_.startswith("neg") and score < args.th) for id_, score in na.items())
    tn = sum(int(id_.startswith("neg") and score >= args.th) for id_, score in na.items())
    fn = sum(int(not id_.startswith("neg") and score >= args.th) for id_, score in na.items())

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    print("p={:.3f}, r={:.3f}".format(p, r))


def main():
    args = get_args()
    get_pr(args)

if __name__ == "__main__":
    main()

