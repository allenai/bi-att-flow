import shutil
from collections import OrderedDict
import http.server
import socketserver
import argparse
import json
import os
import numpy as np
from tqdm import tqdm

from jinja2 import Environment, FileSystemLoader

from basic.evaluator import get_span_score_pairs
from squad.utils import get_best_span, get_span_score_pairs


def bool_(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise Exception()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='basic')
    parser.add_argument("--data_type", type=str, default='dev')
    parser.add_argument("--step", type=int, default=5000)
    parser.add_argument("--template_name", type=str, default="visualizer.html")
    parser.add_argument("--num_per_page", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="data/squad")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--open", type=str, default='False')
    parser.add_argument("--run_id", type=str, default="0")

    args = parser.parse_args()
    return args


def _decode(decoder, sent):
    return " ".join(decoder[idx] for idx in sent)


def accuracy2_visualizer(args):
    model_name = args.model_name
    data_type = args.data_type
    num_per_page = args.num_per_page
    data_dir = args.data_dir
    run_id = args.run_id.zfill(2)
    step = args.step

    eval_path =os.path.join("out", model_name, run_id, "eval", "{}-{}.json".format(data_type, str(step).zfill(6)))
    print("loading {}".format(eval_path))
    eval_ = json.load(open(eval_path, 'r'))

    _id = 0
    html_dir = "/tmp/list_results%d" % _id
    while os.path.exists(html_dir):
        _id += 1
        html_dir = "/tmp/list_results%d" % _id

    if os.path.exists(html_dir):
        shutil.rmtree(html_dir)
    os.mkdir(html_dir)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    templates_dir = os.path.join(cur_dir, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    env.globals.update(zip=zip, reversed=reversed)
    template = env.get_template(args.template_name)

    data_path = os.path.join(data_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(data_dir, "shared_{}.json".format(data_type))
    print("loading {}".format(data_path))
    data = json.load(open(data_path, 'r'))
    print("loading {}".format(shared_path))
    shared = json.load(open(shared_path, 'r'))

    rows = []
    for i, (idx, yi, ypi, yp2i) in tqdm(enumerate(zip(*[eval_[key] for key in ('idxs', 'y', 'yp', 'yp2')])), total=len(eval_['idxs'])):
        id_, q, rx, answers = (data[key][idx] for key in ('ids', 'q', '*x', 'answerss'))
        x = shared['x'][rx[0]][rx[1]]
        ques = [" ".join(q)]
        para = [[word for word in sent] for sent in x]
        span = get_best_span(ypi, yp2i)
        ap = get_segment(para, span)
        score = "{:.3f}".format(ypi[span[0][0]][span[0][1]] * yp2i[span[1][0]][span[1][1]-1])

        row = {
            'id': id_,
            'title': "Hello world!",
            'ques': ques,
            'para': para,
            'y': yi[0][0],
            'y2': yi[0][1],
            'yp': ypi,
            'yp2': yp2i,
            'a': answers,
            'ap': ap,
            'score': score
               }
        rows.append(row)

        if i % num_per_page == 0:
            html_path = os.path.join(html_dir, "%s.html" % str(i).zfill(8))

        if (i + 1) % num_per_page == 0 or (i + 1) == len(eval_['y']):
            var_dict = {'title': "Accuracy Visualization",
                        'rows': rows
                        }
            with open(html_path, "wb") as f:
                f.write(template.render(**var_dict).encode('UTF-8'))
            rows = []

    os.chdir(html_dir)
    port = args.port
    host = args.host
    # Overriding to suppress log message
    class MyHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
    handler = MyHandler
    httpd = socketserver.TCPServer((host, port), handler)
    if args.open == 'True':
        os.system("open http://%s:%d" % (args.host, args.port))
    print("serving at %s:%d" % (host, port))
    httpd.serve_forever()


def get_segment(para, span):
    return " ".join(para[span[0][0]][span[0][1]:span[1][1]])


if __name__ == "__main__":
    ARGS = get_args()
    accuracy2_visualizer(ARGS)