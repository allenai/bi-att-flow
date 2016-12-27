import argparse
import csv
import json
import os


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")

    parser.add_argument("levy_path")
    parser.add_argument("squad_path")
    parser.add_argument("--num", "-n", type=int, default=-1)

    return parser.parse_args()


def levy2squad(args):
    levy_path = args.levy_path
    squad_path = args.squad_path
    num = args.num

    squad = {'data': [{'paragraphs': []}]}
    squad['version'] = '0.1'
    paras = squad['data'][0]['paragraphs']

    with open(levy_path, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for i, each in enumerate(reader):
            rel, ques_temp, ques_arg, sent = each[:4]
            ques = ques_temp.replace('XXX', ques_arg)
            ans_list = each[4:]
            para = {'context': sent, 'qas': [{'question': ques, 'answers': []}]}
            qa = para['qas'][0]
            qa['id'] = str(i)
            for ans in ans_list:
                ans_start = sent.index(ans)
                qa['answers'].append({'text': ans, 'answer_start': ans_start})
            paras.append(para)
            if args.num >= 0 and i + 1 == num:
                break


    with open(squad_path, 'w') as fp:
        json.dump(squad, fp)


def main():
    args = get_args()
    levy2squad(args)

if __name__ == "__main__":
    main()
