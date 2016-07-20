import json

from tqdm import tqdm


def mytqdm(list_, desc="", show=True):
    if show:
        pbar = tqdm(list_)
        pbar.set_description(desc)
        return pbar
    return list_


def json_pretty_dump(obj, fh):
    return json.dump(obj, fh, sort_keys=True, indent=2, separators=(',', ': '))


def _index(l, index):
    if len(index) == 1:
        return l[index[0]]
    return _index(l[index[0]], index[1:])
