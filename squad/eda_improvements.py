import json
import pprint
from difflib import SequenceMatcher
import numpy as np
from scipy import spatial


pp = pprint.PrettyPrinter(indent=2)


def failed_answers():
	with open('/Applications/MAMP/htdocs/bi-att-flow/data/squad/dev-v1.1.json') as f:
		q_and_c = json.load(f)

	with open('/Applications/MAMP/htdocs/bi-att-flow/single.json') as f:
		predicted_answers = json.load(f)

	def similar(a, b):
		return SequenceMatcher(None, a, b).ratio()
	# print("similar {}".format(similar('The premier', 'premier')))

	total_qas, failed_qas = 0, 0
	for article in q_and_c['data'][2:10]:
		print(article.keys())
		for paragraph in article['paragraphs']:
			context = paragraph['context']
			# print("--------------------------------------------------------------------")
			# print("||| context: {}\n num_qas: {}".format(context, len(paragraph['qas'])))
			failed = 0
			for question in paragraph['qas']:
				total_qas += 1
				ground_truth_answers = [ans['text'] for ans in question['answers']]
				# Let's find out why the model fails
				predicted_answer = predicted_answers[question['id']]
				low_similarity = True
				for gta in ground_truth_answers:
					if similar(predicted_answer, gta) > 0.8:
						low_similarity = False

				if predicted_answer not in ground_truth_answers and low_similarity:
					failed_qas += 1
					print("--------------------------------------------------------------------")
					# print("||| context: {}\n num_qas: {}".format(context, len(paragraph['qas'])))
					# print("-----------")
					print('||| context: {}\n ||| question: {} \n||| predicted_answer: {} \n||| ground_truth_answers: {}'.format(
						context,
						question['question'],
						predicted_answers[question['id']],
						ground_truth_answers
					))
					# failed += 1
					# break
			# print("The model has failed to predict {}/{} questions".format(failed, len(paragraph['qas'])))
			# break
		# break
	print("total_qas {} failed_qas {}".format(total_qas, failed_qas))

# failed_answers()


def load_glove_emb():
	word2vec_all = {}
	glove_embedding_size = 100
	with open('/Applications/MAMP/htdocs/bi-att-flow/data/glove/glove.6B.{}d.txt'.format(glove_embedding_size)) as f:
		for line_index, line in enumerate(f):

			parts = line.split()
			word = parts[0]
			embedding_vector = parts[1::]
			word2vec_all[word] = np.array(embedding_vector, dtype=np.float32)
	return word2vec_all

glove_emb = load_glove_emb()
word1 = 'car'
word2 = 'cars'
similarity = 1 - spatial.distance.cosine(glove_emb[word1], glove_emb[word2])
print("word1 {} word2 {} similarity {}".format(word1, word2, similarity))
