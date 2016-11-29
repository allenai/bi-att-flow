import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm

from squad.utils import get_word_span, process_tokens, get_word_idx

def prepro(rxi, question):
    data_type = 'demo'
    out_name='demo'

    import nltk
    sent_tokenize = lambda para: [para] #  nltk.sent_tokenize
    def word_tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    
    qi = word_tokenize(question)
    cqi = [list(qij) for qij in qi]
    yi = [[(0, 0), (0, 0)]]
    cyi = [[0, 0]]
    answers = []
    q.append(qi)
    cq.append(cqi)
    y.append(yi)
    cy.append(cyi)
    rx.append(rxi)
    rcx.append(rxi)
    ids.append(0)
    idxs.append(len(idxs))
    answerss.append(answers)

    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy, '*p': rx,
            'idxs': idxs, 'ids': ids, 'answerss': answerss}
    return data 

if __name__ == "__main__":
    main()
