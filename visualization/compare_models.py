import numpy as np
from collections import Counter
import string
import re
import argparse
import os
import json
import nltk
from matplotlib_venn import venn2
from matplotlib import pyplot as plt


class Question:
    def __init__(self, id, question_text, ground_truth, model_names):
        self.id = id
        self.question_text = self.normalize_answer(question_text)
        self.question_head_ngram = []
        self.question_tokens = nltk.word_tokenize(self.question_text)
        for nc in range(3):
            self.question_head_ngram.append(' '.join(self.question_tokens[0:nc]))
        self.ground_truth = ground_truth
        self.model_names = model_names
        self.em = np.zeros(2)
        self.f1 = np.zeros(2)
        self.answer_text = []

    def add_answers(self, answer_model_1, answer_model_2):
        self.answer_text.append(answer_model_1)
        self.answer_text.append(answer_model_2)
        self.eval()

    def eval(self):
        for model_count in range(2):
            self.em[model_count] = self.metric_max_over_ground_truths(self.exact_match_score, self.answer_text[model_count], self.ground_truth)
            self.f1[model_count] = self.metric_max_over_ground_truths(self.f1_score, self.answer_text[model_count], self.ground_truth)

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self, prediction, ground_truth):
        return (self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)


def safe_dict_access(in_dict, in_key, default_string='some junk string'):
    if in_key in in_dict:
        return in_dict[in_key]
    else:
        return default_string


def aggregate_metrics(questions):
    total = len(questions)
    exact_match = np.zeros(2)
    f1_scores = np.zeros(2)

    for mc in range(2):
        exact_match[mc] = 100 * np.sum(np.array([questions[x].em[mc] for x in questions])) / total
        f1_scores[mc] = 100 * np.sum(np.array([questions[x].f1[mc] for x in questions])) / total

    model_names = questions[list(questions.keys())[0]].model_names
    print('\nAggregate Scores:')
    for model_count in range(2):
        print('Model {0} EM = {1:.2f}'.format(model_names[model_count], exact_match[model_count]))
        print('Model {0} F1 = {1:.2f}'.format(model_names[model_count], f1_scores[model_count]))


def venn_diagram(questions, output_dir):
    em_model1_ids = [x for x in questions if questions[x].em[0] == 1]
    em_model2_ids = [x for x in questions if questions[x].em[1] == 1]
    model_names = questions[list(questions.keys())[0]].model_names
    print('\nVenn diagram')

    correct_model1 = em_model1_ids
    correct_model2 = em_model2_ids
    correct_model1_and_model2 = list(set(em_model1_ids).intersection(set(em_model2_ids)))
    correct_model1_and_not_model2 = list(set(em_model1_ids) - set(em_model2_ids))
    correct_model2_and_not_model1 = list(set(em_model2_ids) - set(em_model1_ids))

    print('{0} answers correctly = {1}'.format(model_names[0], len(correct_model1)))
    print('{0} answers correctly = {1}'.format(model_names[1], len(correct_model2)))
    print('Both answer correctly = {1}'.format(model_names[0], len(correct_model1_and_model2)))
    print('{0} correct & {1} incorrect = {2}'.format(model_names[0], model_names[1], len(correct_model1_and_not_model2)))
    print('{0} correct & {1} incorrect = {2}'.format(model_names[1], model_names[0], len(correct_model2_and_not_model1)))

    plt.clf()
    venn_diagram_plot = venn2(
        subsets=(len(correct_model1_and_not_model2), len(correct_model2_and_not_model1), len(correct_model1_and_model2)),
        set_labels=('{0} correct'.format(model_names[0]), '{0} correct'.format(model_names[1]), 'Both correct'),
        set_colors=('r', 'b'),
        alpha=0.3,
        normalize_to=1
    )
    plt.savefig(os.path.join(output_dir, 'venn_diagram.png'))
    plt.close()
    return correct_model1, correct_model2, correct_model1_and_model2, correct_model1_and_not_model2, correct_model2_and_not_model1


def get_head_ngrams(questions, num_grams):
    head_ngrams = []
    for question in questions.values():
        head_ngrams.append(question.question_head_ngram[num_grams])
    return head_ngrams


def get_head_ngram_frequencies(questions, head_ngrams, num_grams):
    head_ngram_frequencies = {}
    for current_ngram in head_ngrams:
        head_ngram_frequencies[current_ngram] = 0
    for question in questions.values():
        head_ngram_frequencies[question.question_head_ngram[num_grams]] += 1
    return head_ngram_frequencies


def get_head_ngram_statistics(questions, correct_model1, correct_model2, correct_model1_and_model2, correct_model1_and_not_model2, correct_model2_and_not_model1, output_dir, num_grams=2, top_count=25):
    # Head ngram statistics
    head_ngrams = get_head_ngrams(questions, num_grams)

    # Get head_ngram_frequencies (hnf)
    hnf_all = get_head_ngram_frequencies(questions, head_ngrams, num_grams)
    hnf_correct_model1 = get_head_ngram_frequencies({qid: questions[qid] for qid in correct_model1}, head_ngrams, num_grams)
    hnf_correct_model2 = get_head_ngram_frequencies({qid: questions[qid] for qid in correct_model2}, head_ngrams, num_grams)
    hnf_correct_model1_and_model2 = get_head_ngram_frequencies({qid: questions[qid] for qid in correct_model1_and_model2}, head_ngrams, num_grams)
    hnf_correct_model1_and_not_model2 = get_head_ngram_frequencies({qid: questions[qid] for qid in correct_model1_and_not_model2}, head_ngrams, num_grams)
    hnf_correct_model2_and_not_model1 = get_head_ngram_frequencies({qid: questions[qid] for qid in correct_model2_and_not_model1}, head_ngrams, num_grams)

    sorted_bigrams_all = sorted(hnf_all.items(), key=lambda x: x[1], reverse=True)
    top_bigrams = [x[0] for x in sorted_bigrams_all[0:top_count]]

    counts_total = [hnf_all[x] for x in top_bigrams]
    counts_model1 = [hnf_correct_model1[x] for x in top_bigrams]
    counts_model2 = [hnf_correct_model2[x] for x in top_bigrams]
    counts_model1_and_model2 = [hnf_correct_model1_and_model2[x] for x in top_bigrams]
    counts_model1_and_not_model2 = [hnf_correct_model1_and_not_model2[x] for x in top_bigrams]
    counts_model2_and_not_model1 = [hnf_correct_model2_and_not_model1[x] for x in top_bigrams]

    top_bigrams_with_counts = []
    for cc in range(len(top_bigrams)):
        top_bigrams_with_counts.append('{0} ({1})'.format(top_bigrams[cc], counts_total[cc]))

    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 10))

    ylocs = list(range(top_count))
    counts_model1_percent = 100 * np.array(counts_model1) / np.array(counts_total)
    plt.barh([top_count - x for x in ylocs], counts_model1_percent, height=0.4, alpha=0.5, color='#EE3224', label=top_bigrams)
    counts_model2_percent = 100 * np.array(counts_model2) / np.array(counts_total)
    plt.barh([top_count - x+0.4 for x in ylocs], counts_model2_percent, height=0.4, alpha=0.5, color='#2432EE', label=top_bigrams  )
    ax.set_yticks([top_count - x + 0.4 for x in ylocs])
    ax.set_yticklabels(top_bigrams_with_counts)
    ax.set_ylim([0.5, top_count+1])
    ax.set_xlim([0, 100])
    plt.subplots_adjust(left=0.28, right=0.9, top=0.9, bottom=0.1)
    plt.xlabel('Percentage of questions with correct answers')
    plt.ylabel('Top N-grams')
    plt.savefig(os.path.join(output_dir, 'ngram_stats_{0}.png'.format(num_grams)))
    plt.close()


def read_json(filename):
    with open(filename) as filepoint:
        data = json.load(filepoint)
    return data


def compare_models(dataset_file, predictions_m1_file, predictions_m2_file, output_dir, name_m1='Model 1', name_m2='Model 2'):
    dataset = read_json(dataset_file)['data']
    predictions_m1 = read_json(predictions_m1_file)
    predictions_m2 = read_json(predictions_m2_file)

    # Read in data
    total = 0
    questions = {}
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                current_question = Question(id=qa['id'], question_text=qa['question'], ground_truth=list(map(lambda x: x['text'], qa['answers'])), model_names=[name_m1, name_m2])
                current_question.add_answers(answer_model_1=safe_dict_access(predictions_m1, qa['id']), answer_model_2=safe_dict_access(predictions_m2, qa['id']))
                questions[current_question.id] = current_question
                total += 1
    model_names = questions[list(questions.keys())[0]].model_names
    print('Read in {0} questions'.format(total))

    # Aggregate scores
    aggregate_metrics(questions)

    # Venn diagram
    correct_model1, correct_model2, correct_model1_and_model2, correct_model1_and_not_model2, correct_model2_and_not_model1 = venn_diagram(questions, output_dir=output_dir)

    # Head Unigram statistics
    get_head_ngram_statistics(questions, correct_model1, correct_model2, correct_model1_and_model2, correct_model1_and_not_model2,
                              correct_model2_and_not_model1, output_dir, num_grams=1, top_count=10)

    # Head Bigram statistics
    get_head_ngram_statistics(questions, correct_model1, correct_model2, correct_model1_and_model2, correct_model1_and_not_model2,
                              correct_model2_and_not_model1, output_dir, num_grams=2, top_count=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two QA models')
    parser.add_argument('-dataset', action='store', dest='dataset', required=True, help='Dataset file')
    parser.add_argument('-model1', action='store', dest='predictions_m1', required=True, help='Prediction file for model 1')
    parser.add_argument('-model2', action='store', dest='predictions_m2', required=True, help='Prediction file for model 2')
    parser.add_argument('-name1', action='store', dest='name_m1', help='Name for model 1')
    parser.add_argument('-name2', action='store', dest='name_m2', help='Name for model 2')
    parser.add_argument('-output', action='store', dest='output_dir', help='Output directory for visualizations')
    results = parser.parse_args()

    if results.name_m1 is not None and results.name_m2 is not None:
        compare_models(dataset_file=results.dataset, predictions_m1_file=results.predictions_m1, predictions_m2_file=results.predictions_m2, output_dir=results.output_dir, name_m1=results.name_m1, name_m2=results.name_m2)
    else:
        compare_models(dataset_file=results.dataset, predictions_m1_file=results.predictions_m1, predictions_m2_file=results.predictions_m2, output_dir=results.output_dir)
