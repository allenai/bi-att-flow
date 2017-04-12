import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm

from squad.utils import get_word_span, get_word_idx, process_tokens


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/squad"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, "train-v1.1.json")
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, "dev-v1.1.json")
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.mode == 'full':
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')
        prepro_each(args, 'dev', out_name='test')
    elif args.mode == 'all':
        create_all(args)
        prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
        prepro_each(args, 'dev', 0.0, 0.0, out_name='test')
        prepro_each(args, 'all', out_name='train')
    elif args.mode == 'dev':
        prepro_each(args, 'dev', out_name='dev')
    elif args.mode == 'dev_short':
        prepro_each(args, 'dev_short', 0.0, 0.05, out_name='dev_short')
    elif args.mode == 'single':
        assert len(args.single_path) > 0
        prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    else:
        prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
        prepro_each(args, 'train', args.train_ratio, 1.0, out_name='dev')
        prepro_each(args, 'dev', out_name='test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_single_question_with_context(context, question):

    import nltk
    sent_tokenize = nltk.sent_tokenize

    def word_tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

    qi = word_tokenize(question)
    cqi = [list(qij) for qij in qi]
    yi0 = None
    yi1 = None
    y = [[yi0, yi1]]
    ai, pi = 0, 0
    rxi = [ai, pi]
    rx = [rxi]
    rcx = rx
    cy = rxi
    idxs = [2]
    ids = ['56be4db0acb8001400a502ee']
    answerss = [['Carrots']]

    # context = para['context']
    context = context.replace("''", '" ')
    context = context.replace("``", '" ')
    xi = word_tokenize(context)
    # Transform list of words into list of characters
    cxi = [list(xij) for xij in xi]

    data = {
        'q': [qi],                 # raw questions
        'cq': [cqi],               # raw questions characters
        'y': [[[[0, 0], [0, 0]]]],
        '*x': [[0, 0]],
        '*cx': [[0, 0]],
        'cy': [[[0, 0]]],
        'idxs': [2],
        'idxs': idxs,           # Useless??
        'ids': ids,             # question ids
        'answerss': answerss,   # raw answers
        '*p': rx,                # *x Useless?
        'x': [[xi]],
        'cx': [[cxi]],
        'p': [context]
    }
    # pprint(data)
    return data

"""
    Preprocess each dataset
"""
def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "{}-v1.1.json".format(data_type))
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    # Iterate through articles
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp = [], []
        pp = []
        x.append(xp)
        cx.append(cxp)
        p.append(pp)
        # Iterate through paragraphs
        for pi, para in enumerate(article['paragraphs']):
            # Words
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            # Split paragraphs into sentences and then split into words.
            # I like carrots. Very much. => ["I", "like", "carrots", ".", "Very", "much", "."]
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # Remove non-ascii characters. TODO: Encode to utf-8
            # Transform list of words into list of characters
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            # Append to buckets
            xp.append(xi) # append all the words to xp
            cxp.append(cxi) # append all the characters to cxp
            pp.append(context) # Store raw text

            # Count the number of words and characters used across all the articles
            # TODO: not sure why not use xi[0] as len(xi) is always 1
            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            # Process questions and answers
            for qa in para['qas']:
                # Split question into words (qi) and then split into chars (cqi).
                qi = word_tokenize(qa['question'])
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                # Iterate through answers
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text) # Store raw answer
                    # Get start and end indices, which will eventually be the targets
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)

                    # Get word the word start and end instead of char start and end.
                    # I'm not sure why is this useful. It would be for CNN
                    # TODO : put some function that gives word_start, word_stop here
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    w0 = xi[yi0[0]][yi0[1]] # Get start word
                    w1 = xi[yi1[0]][yi1[1]-1] # Get end word

                    i0 = get_word_idx(context, xi, yi0) # Get word start index from the context
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1]-1)) # Get word end index from the context

                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])

                # Count words and characters in the question
                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1

                # Append all the stuff into the lists
                q.append(qi)
                cq.append(cqi)
                y.append(yi)
                cy.append(cyi)
                rx.append(rxi)
                rcx.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                answerss.append(answers)

            if args.debug:
                break
        # break

    print('...get_word2vec (word2vec_dict)')
    word2vec_dict = get_word2vec(args, word_counter)

    print('...get_word2vec (lower_word2vec_dict)')
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    data = {
        'q': q,                 # raw questions
        'cq': cq,               # raw questions characters
        'y': y,                 # word answers spans
        '*x': rx,               # question answer indexes
        '*cx': rcx,             # *x ?¿?¿?
        'cy': cy,               # answer in words
        'idxs': idxs,           # Useless??
        'ids': ids,             # question ids
        'answerss': answerss,   # raw answers
        '*p': rx                # *x Useless?
    }

    shared = {
        'x': x,                                     # all the articles as list of words
        'cx': cx,                                   # all the articles as a list of characters
        'p': p,                                     # raw context/paragraphs
        'word_counter': word_counter,               # self explanatory
        'char_counter': char_counter,               # self explanatory
        'lower_word_counter': lower_word_counter,   # lower case word counter
        'word2vec': word2vec_dict,                  # word to vector dictionary of all the present words
        'lower_word2vec': lower_word2vec_dict       # same as word2vec but lower case
    }

    print("saving ...")
    save(args, data, shared, out_name)



if __name__ == "__main__":
    main()
    # context = 'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.'
    # question = 'Where did Super Bowl 50 take place?'

    # context = 'Carrots are very evil creatures. I know a lot of creatures like carrots.'
    # question = 'What are carrots?'
    # # prepro_single_question_with_context('The carrot was born in August.', 'When was the carrot born?')
    # pprint(prepro_single_question_with_context(context, question))
