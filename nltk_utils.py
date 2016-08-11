import nltk


def _set_span(t, i):
    if isinstance(t[0], str):
        t.span = (i, i+len(t))
    else:
        first = True
        for c in t:
            cur_span = _set_span(c, i)
            i = cur_span[1]
            if first:
                min_ = cur_span[0]
                first = False
        max_ = cur_span[1]
        t.span = (min_, max_)
    return t.span


def set_span(t):
    assert isinstance(t, nltk.tree.Tree)
    try:
        return _set_span(t, 0)
    except:
        print(t)
        exit()


def tree_contains_span(tree, span):
    """
    Assumes that tree span has been set with set_span
    Returns true if any subtree of t has exact span as the given span
    :param t:
    :param span:
    :return bool:
    """
    return span in set(t.span for t in tree.subtrees())
