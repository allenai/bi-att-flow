import json
import sys

from tqdm import tqdm

from my.corenlp_interface import CoreNLPInterface

in_path = sys.argv[1]
out_path = sys.argv[2]
url = sys.argv[3]
port = int(sys.argv[4])
data = json.load(open(in_path, 'r'))

h = CoreNLPInterface(url, port)


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def to_hex(s):
    return " ".join(map(hex, map(ord, s)))


def handle_nobreak(cand, text):
    if cand == text:
        return cand
    if cand.replace(u'\u00A0', ' ') == text:
        return cand
    elif cand == text.replace(u'\u00A0', ' '):
        return text
    raise Exception("{} '{}' {} '{}'".format(cand, to_hex(cand), text, to_hex(text)))


# resolving unicode complication

wrong_loc_count = 0
loc_diffs = []

for article in data['data']:
    for para in article['paragraphs']:
        para['context'] = para['context'].replace(u'\u000A', '')
        para['context'] = para['context'].replace(u'\u00A0', ' ')
        context = para['context']
        for qa in para['qas']:
            for answer in qa['answers']:
                answer['text'] = answer['text'].replace(u'\u00A0', ' ')
                text = answer['text']
                answer_start = answer['answer_start']
                if context[answer_start:answer_start + len(text)] == text:
                    if text.lstrip() == text:
                        pass
                    else:
                        answer_start += len(text) - len(text.lstrip())
                        answer['answer_start'] = answer_start
                        text = text.lstrip()
                        answer['text'] = text
                else:
                    wrong_loc_count += 1
                    text = text.lstrip()
                    answer['text'] = text
                    starts = list(find_all(context, text))
                    if len(starts) == 1:
                        answer_start = starts[0]
                    elif len(starts) > 1:
                        new_answer_start = min(starts, key=lambda s: abs(s - answer_start))
                        loc_diffs.append(abs(new_answer_start - answer_start))
                        answer_start = new_answer_start
                    else:
                        raise Exception()
                    answer['answer_start'] = answer_start

                answer_stop = answer_start + len(text)
                answer['answer_stop'] = answer_stop
                assert para['context'][answer_start:answer_stop] == answer['text'], "{} {}".format(
                    para['context'][answer_start:answer_stop], answer['text'])

print(wrong_loc_count, loc_diffs)

mismatch_count = 0
dep_fail_count = 0
no_answer_count = 0

size = sum(len(article['paragraphs']) for article in data['data'])
pbar = tqdm(range(size))

for ai, article in enumerate(data['data']):
    for pi, para in enumerate(article['paragraphs']):
        context = para['context']
        sents = h.split_doc(context)
        words = h.split_sent(context)
        sent_starts = []
        ref_idx = 0
        for sent in sents:
            new_idx = context.find(sent, ref_idx)
            sent_starts.append(new_idx)
            ref_idx = new_idx + len(sent)
        para['sents'] = sents
        para['words'] = words
        para['sent_starts'] = sent_starts

        consts = list(map(h.get_const, sents))
        para['consts'] = consts
        deps = list(map(h.get_dep, sents))
        para['deps'] = deps

        for qa in para['qas']:
            question = qa['question']
            question_const = h.get_const(question)
            qa['const'] = question_const
            question_dep = h.get_dep(question)
            qa['dep'] = question_dep
            qa['words'] = h.split_sent(question)

            for answer in qa['answers']:
                answer_start = answer['answer_start']
                text = answer['text']
                answer_stop = answer_start + len(text)
                # answer_words = h.split_sent(text)
                word_idxs = []
                answer_words = []
                for sent_idx, (sent, sent_start, dep) in enumerate(zip(sents, sent_starts, deps)):
                    if dep is None:
                        print("dep parse failed at {} {} {}".format(ai, pi, sent_idx))
                        dep_fail_count += 1
                        continue
                    nodes, edges = dep
                    words = [node[0] for node in nodes]

                    for word_idx, (word, _, _, start, _) in enumerate(nodes):
                        global_start = sent_start + start
                        global_stop = global_start + len(word)
                        if answer_start <= global_start < answer_stop or answer_start < global_stop <= answer_stop:
                            word_idxs.append((sent_idx, word_idx))
                            answer_words.append(word)
                if len(word_idxs) > 0:
                    answer['answer_word_start'] = word_idxs[0]
                    answer['answer_word_stop'] = word_idxs[-1][0], word_idxs[-1][1] + 1
                    if not text.startswith(answer_words[0]):
                        print("'{}' '{}'".format(text, ' '.join(answer_words)))
                        mismatch_count += 1
                else:
                    answer['answer_word_start'] = None
                    answer['answer_word_stop'] = None
                    no_answer_count += 1
        pbar.update(1)
pbar.close()

print(mismatch_count, dep_fail_count, no_answer_count)

print("saving...")
json.dump(data, open(out_path, 'w'))