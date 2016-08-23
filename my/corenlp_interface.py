import logging

import requests
import nltk
import json
import networkx as nx
import time


class CoreNLPInterface(object):
    def __init__(self, url, port):
        self._url = url
        self._port = port

    def get(self, type_, in_, num_max_requests=100):
        in_ = in_.encode("utf-8")
        url = "http://{}:{}/{}".format(self._url, self._port, type_)
        out = None
        for _ in range(num_max_requests):
            try:
                r = requests.post(url, data=in_)
                out = r.content.decode('utf-8')
                if out == 'error':
                    out = None
                break
            except:
                time.sleep(1)
        return out

    def split_doc(self, doc):
        out = self.get("doc", doc)
        return out if out is None else json.loads(out)

    def split_sent(self, sent):
        out = self.get("sent", sent)
        return out if out is None else json.loads(out)

    def get_dep(self, sent):
        out = self.get("dep", sent)
        return out if out is None else json.loads(out)

    def get_const(self, sent):
        out = self.get("const", sent)
        return out

    def get_const_tree(self, sent):
        out = self.get_const(sent)
        return out if out is None else nltk.tree.Tree.fromstring(out)

    @staticmethod
    def dep2tree(dep):
        tree = nx.DiGraph()
        for dep, i, gov, j, label in dep:
            tree.add_edge(gov, dep, label=label)
        return tree
