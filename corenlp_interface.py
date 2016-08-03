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
        return None if out == "error" else out

    def split_doc(self, doc):
        out = self.get("doc", doc)
        if out is None:
            logging.error("Error occurred during doc_split: {}".format(doc))
            return None
        return json.loads(out)

    def split_sent(self, sent):
        out = self.get("sent", sent)
        if out is None:
            logging.error("Error occurred during sent_split: {}".format(sent))
            return None
        return json.loads(out)

    def get_dep(self, sent):
        out = self.get("dep", sent)
        if out is None:
            logging.error("Error occurred during dep: {}".format(sent))
            return None
        return json.loads(out)

    def get_const_str(self, sent):
        out = self.get("const", sent)
        if out is None:
            logging.error("Error occurred during const: {}".format(sent))
            return None
        return out

    def get_const(self, sent):
        out = self.get_const_str(sent)
        if out is None:
            return None
        return nltk.tree.Tree.fromstring(out)