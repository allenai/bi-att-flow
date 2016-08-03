import logging

import requests
import nltk
import json


class CoreNLPInterface(object):
    def __init__(self, url, port):
        self._url = url
        self._port = port

    def get(self, type_, in_):
        in_ = in_.encode("utf-8")
        url = "http://{}:{}/{}".format(self._url, self._port, type_)
        r = requests.post(url, data=in_)
        out = r.text
        return out

    def split_doc(self, doc):
        out = self.get("doc", doc)
        return json.loads(out)

    def split_sent(self, sent):
        out = self.get("sent", sent)
        return json.loads(out)

    def get_dep(self, sent):
        out = self.get("dep", sent)
        return json.loads(out)

    def get_const_str(self, sent):
        out = self.get("const", sent)
        return out

    def get_const(self, sent):
        out = self.get_const_str(sent)
        return nltk.tree.Tree.fromstring(out)
