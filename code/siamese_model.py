from preprocess import *
from siamese import *

import os, sys
import pickle
import numpy as np
import tensorflow as tf

DEFAULT_SENT_SIZE = 25
DEFAULT_WORD = '駧'

class SiameseModel:

	def __init__(self):
		# add some member
		self.words = []
		self.embeddings = []
		self.word_id = []
		self.id_word = []
		self.num_epochs = 5
		self.test_gap = 100
		self.save_gap = 1000

	def load_previous_model(self, sess):
		load = False
		model_ckpt = '../model/model.meta'
		if os.path.isfile(model_ckpt):
		    input_var = None
		    while input_var not in ['yes', 'no']:
		        input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
		    if input_var == 'yes':
		        load = True

		# start training
		if load: self.saver.restore(sess, './model')

	def load_embedding(self, embed_path):

		if not embed_path:
			exit("ERROR: path to embedding file is invalid!")

		words, embeddings = pickle.load(open("../embed/polyglot-zh.pkl", 'rb'), encoding='iso-8859-1')

		# Map words to indices and vice versa
		self.word_id = {w:i for (i, w) in enumerate(words)}
		num_pretrained = len(self.word_id) # get the number of total pretrained words

		self.word_id[DEFAULT_WORD] = num_pretrained
		self.id_word = dict(enumerate(words))
		self.id_word[num_pretrained] = DEFAULT_WORD
		self.embeddings = np.append(embeddings, [np.zeros(len(embeddings[0]))], axis = 0) # add default value


	def map_sentence_to_id(self, sentences):
		id_batch = []
		DEFAULT_ID = len(self.id_word) - 1
		for s in sentences:
			cur_sent = []
			for w in s:
				#print(w)
				if w not in self.word_id:
					cur_sent.append(DEFAULT_ID)
				else:
					cur_sent.append(self.word_id[w])
			id_batch.append(cur_sent)
		return id_batch

	def network_setup(self):
		# setup siamese network
		self.siamese_net = siamese(DEFAULT_SENT_SIZE, self.embeddings, self.word_id);
		self.train_step = tf.train.AdamOptimizer().minimize(self.siamese_net.loss)
		self.saver = tf.train.Saver()
		init_op = tf.global_variables_initializer()
		return init_op

	def train_on_batch(self, sess, data_x1, data_x2, data_y, cur_iter):

		_, loss = sess.run([self.train_step, self.siamese_net.loss], feed_dict={
                        self.siamese_net.x1: data_x1,
                        self.siamese_net.x2: data_x2,
                        self.siamese_net.y: data_y})
		if cur_iter % 10 == 0:
			print ('step %d: loss %.3f' % (cur_iter, loss))


	def run(self):

		print("######## Now load embeddings ########")

		# first load in pretrained embedding
		self.load_embedding("../embed/polyglot-zh.pkl")

		print("######## Now load data ########")
		# then load in provided data, and split into train and test set
		data_list = read_csv("../data/atec_nlp_sim_train.csv")
		train_data, test_data = prepare_train_test(data_list)

		train_data_x1, train_data_x2, train_data_y, train_mask_x1, train_mask_x2 = load_batches(train_data, batch_size=32) # load training data into batches
		self.test_data_x1, self.test_data_x2, self.test_data_y, self.test_mask_x1, self.test_mask_x2  = load_batches(test_data, batch_size=len(test_data)) # transform testing data into easy form 
		
		#print(train_data_x2)
		self.test_data_x1 = self.map_sentence_to_id(self.test_data_x1[0])
		self.test_data_x2 = self.map_sentence_to_id(self.test_data_x2[0])
		self.test_data_y = self.test_data_y[0]

		print("######## Model initialization ########")
		# initialize the model
		init_op = self.network_setup()

		sess = tf.Session() # open a new session
		sess.run(init_op)
		self.load_previous_model(sess) # ask if want to load a pre-trained model


		print("######## Here we go! ########")
		num_batches = len(train_data_x1) 
		cur_iter = 0
		for epoch in range(self.num_epochs):
			# each epoch
			for i in range(num_batches):
				# each iteration

				self.train_on_batch(sess, 
									self.map_sentence_to_id(train_data_x1[i]), 
									self.map_sentence_to_id(train_data_x2[i]), 
									train_data_y[i], 
									cur_iter)
				cur_iter += 1
				if cur_iter % self.test_gap == 0:
					self.evaluate(sess)
				if cur_iter % self.save_gap == 0:
					self.saver.save("../model/model")

	def evaluate(self, sess):
		# first get the cosine similarity of each pair in testing set
		cosine_sim = sess.run([self.siamese_net.cosine_sim], feed_dict={
                        self.siamese_net.x1: self.test_data_x1,
                        self.siamese_net.x2: self.test_data_x2,
                        self.siamese_net.y: self.test_data_y})
		cosine_sim = np.array(cosine_sim)
		# second round the similarity to the nearest integer
		# 0: not the same
		# 1: same semantic
		threshold = 0.5 # default threshold 0.5
		judge_res = np.ceil(cosine_sim * (cosine_sim > threshold))

		accuracy = np.sum(judge_res == self.test_data_y) * 1.0 / len(self.test_data_y)
		print("\nAccuracy on validation set: " + str(accuracy) + "\n")


if __name__ == "__main__":

	siamese_model = SiameseModel() # create a model instance 
	siamese_model.run()
	





