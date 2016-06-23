import json
import os
from collections import OrderedDict
from copy import deepcopy

from config.tsv2json import tsv2dict


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config(d0, d1, priority=1):
    """
    d1 replaces d0. If priority = 0, then d0 replaces d1
    :param d0:
    :param d1:
    :param name:
    :param priority:
    :return:
    """
    if priority == 0:
        d0, d1 = d1, d0
    d = deepcopy(d0)
    for key, val in d1.items():
        if val is not None:
            d[key] = val
    return Config(**d)


def get_config_from_file(d0, path, id_, priority=1):
    _, ext = os.path.splitext(path)
    if ext == '.json':
        configs = json.load(open(path, 'r'), object_pairs_hook=OrderedDict)
    elif ext == '.tsv':
        configs = tsv2dict(path)
    else:
        raise Exception("Extension %r is not supported." % ext)
    d1 = configs[id_]
    return get_config(d0, d1, priority=priority)





