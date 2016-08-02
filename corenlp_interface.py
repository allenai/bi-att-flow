import requests
import nltk
import json


class CoreNLPInterface(object):
    def __init__(self, url, port):
        self._url = url
        self._port = port

    def get(self, type_, in_):
        url = "http://{}:{}/{}".format(self._url, self._port, type_)
        r = requests.post(url, data=in_)
        return r.text

    def split_doc(self, doc):
        return json.loads(self.get("doc", doc))

    def split_sent(self, sent):
        return json.loads(self.get("sent", sent))

    def get_dep(self, sent):
        return json.loads(self.get("dep", sent))

    def get_const(self, sent):
        return nltk.tree.Tree.fromstring(self.get("const", sent))