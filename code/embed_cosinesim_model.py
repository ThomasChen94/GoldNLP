from preprocess import read_csv, prepare_train_test

import pickle
import numpy


class Embed_CosineSim_Model:

	def __init__():
		# add some member
		self.words = []
		self.embeddings = []
		self.word_id = []
		self.id_word = []
		self.num_epochs = 5

	def load_embedding(embed_path):

		if not embed_path:
			exit("ERROR: path to embedding file is invalid!")
		self.words, self.embeddings = pickle.load(open(embed_path, 'rb'))

		# Map words to indices and vice versa
		self.word_id = {w:i for (i, w) in enumerate(words)}
		self.id_word = dict(enumerate(words))


	def train():
		for epoch in xrange(num_epochs):



	if __name__ == "__main__":

		# first load in pretrained embedding
		embeddings, word_id, id_word  = load_embedding("../embed/polyglot-zh.pkl")
		data_list = read_csv("../data/atec_nlp_sim_train.csv")
		train_data, test_data = prepare_train_test(data_list)
		if data_list[0][0][1] in word_id:
			print embeddings[word_id[data_list[0][0][1]]]

