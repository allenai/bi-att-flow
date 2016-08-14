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


def span_len(span):
    return span[1] - span[0]


def span_overlap(s1, s2):
    start = max(s1[0], s2[0])
    stop = min(s1[1], s2[1])
    if stop > start:
        return start, stop
    return None


def span_prec(true_span, pred_span):
    overlap = span_overlap(true_span, pred_span)
    overlap_size = 0 if overlap is None else span_len(overlap)
    return overlap_size / span_len(pred_span)


def span_recall(true_span, pred_span):
    overlap = span_overlap(true_span, pred_span)
    overlap_size = 0 if overlap is None else span_len(overlap)
    return overlap_size / span_len(true_span)


def span_f1(true_span, pred_span):
    p = span_prec(true_span, pred_span)
    r = span_recall(true_span, pred_span)
    if p == 0 or r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def find_max_f1_span(tree, span):
    return max(((t.span, span_f1(t.span, span)) for t in tree.subtrees()), key=lambda p: p[1])


