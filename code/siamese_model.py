from preprocess import *
from siamese import *

import pickle
import numpy
import tensorflow as tf

class SiameseModel:

	def __init__():
		# add some member
		self.words = []
		self.embeddings = []
		self.word_id = []
		self.id_word = []
		self.num_epochs = 5
		self.test_gap = 100

	def load_previous_model(sess):
		load = False
		model_ckpt = '../model/model.meta'
		if os.path.isfile(model_ckpt):
		    input_var = None
		    while input_var not in ['yes', 'no']:
		        input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
		    if input_var == 'yes':
		        load = True

		# start training
		if load: saver.restore(sess, './model')

	def load_embedding(embed_path):

		if not embed_path:
			exit("ERROR: path to embedding file is invalid!")
		self.words, self.embeddings = pickle.load(open(embed_path, 'rb'))

		# Map words to indices and vice versa
		self.word_id = {w:i for (i, w) in enumerate(words)}
		self.id_word = dict(enumerate(words))

	def network_setup():
		# setup siamese network
		self.siamese_net = siamese();
		self.train_step = tf.train.AdamOptimizer().minimize(siamese.loss)
		self.saver = tf.train.Saver()
		tf.initialize_all_variables().run()


	def train_on_batch(data_x1, data_x2, data_y, cur_iter):
		_, loss = sess.run([self.train_step, self.siamese_net.loss], feed_dict={
                        siamese.x1: data_x1,
                        siamese.x2: data_x2,
                        siamese.y: data_y})
		if cur_iter % 10 == 0:
			print ('step %d: loss %.3f' % (step, loss))


	def run():
		# first load in pretrained embedding
		embeddings, word_id, id_word  = load_embedding("../embed/polyglot-zh.pkl")

		# then load in provided data, and split into train and test set
		data_list = read_csv("../data/atec_nlp_sim_train.csv")
		train_data, test_data = prepare_train_test(data_list)

		# initialize the model
		siamese_net, train_step, saver = network_setup()

		sess = tf.session() # open a new session
		load_previous_model(sess) # ask if want to load a pre-trained model

		train_data_x1, train_data_x2, train_data_y = load_batches(train_data, batch_size=32) # load training data into batches
		self.test_data_x1, self.test_data_x2, self.test_data_y = load_batches(test_data, batch_size=len(test_data)) # transform testing data into easy form 

		num_batches = len(train_data_x1) 
		cur_iter = 0
		for epoch in xrange(self.num_epochs):
			# each epoch
			for i in xrange(num_batches):
				# each iteration
				train_on_batch(data_x1[i], data_x2[i], data_y[i], cur_iter)
				cur_iter += 1
				if cur_iter % self.test_gap == 0:
					evaluate()
				if cur_iter % self.save_gap == 0:
					saver.save(../model/model)

	def evaluate():
		# first get the cosine similarity of each pair in testing set
		cosine_sim = sess.run([self.siamese_net.cosine_sim], feed_dict={
                        siamese.x1: self.test_data_x1,
                        siamese.x2: self.test_data_x2,
                        siamese.y: self.test_data_y})
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
	





