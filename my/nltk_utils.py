import nltk
import numpy as np


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
    if overlap is None:
        return 0
    return span_len(overlap) / span_len(pred_span)


def span_recall(true_span, pred_span):
    overlap = span_overlap(true_span, pred_span)
    if overlap is None:
        return 0
    return span_len(overlap) / span_len(true_span)


def span_f1(true_span, pred_span):
    p = span_prec(true_span, pred_span)
    r = span_recall(true_span, pred_span)
    if p == 0 or r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def find_max_f1_span(tree, span):
    return find_max_f1_subtree(tree, span).span


def find_max_f1_subtree(tree, span):
    return max(((t, span_f1(span, t.span)) for t in tree.subtrees()), key=lambda p: p[1])[0]


def tree2matrix(tree, node2num, row_size=None, col_size=None, dtype='int32'):
    set_span(tree)
    D = tree.height() - 1
    B = len(tree.leaves())
    row_size = row_size or D
    col_size = col_size or B
    matrix = np.zeros([row_size, col_size], dtype=dtype)
    mask = np.zeros([row_size, col_size, col_size], dtype='bool')

    for subtree in tree.subtrees():
        row = subtree.height() - 2
        col = subtree.span[0]
        matrix[row, col] = node2num(subtree)
        for subsub in subtree.subtrees():
            if isinstance(subsub, nltk.tree.Tree):
                mask[row, col, subsub.span[0]] = True
                if not isinstance(subsub[0], nltk.tree.Tree):
                    c = subsub.span[0]
                    for r in range(row):
                        mask[r, c, c] = True
            else:
                mask[row, col, col] = True

    return matrix, mask


def load_compressed_tree(s):

    def compress_tree(tree):
        assert not isinstance(tree, str)
        if len(tree) == 1:
            if isinstance(tree[0], nltk.tree.Tree):
                return compress_tree(tree[0])
            else:
                return tree
        else:
            for i, t in enumerate(tree):
                if isinstance(t, nltk.tree.Tree):
                    tree[i] = compress_tree(t)
                else:
                    tree[i] = t
            return tree

    return compress_tree(nltk.tree.Tree.fromstring(s))



