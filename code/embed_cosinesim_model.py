from preprocess import *
from siamese import *

import pickle
import numpy
import tensorflow as tf

class Embed_CosineSim_Model:

	def __init__():
		# add some member
		self.words = []
		self.embeddings = []
		self.word_id = []
		self.id_word = []
		self.num_epochs = 5

	def load_previous_model(sess):
		load = False
		model_ckpt = './model.meta'
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
		siamese_net = siamese();
		train_step = tf.train.AdamOptimizer().minimize(siamese.loss)
		saver = tf.train.Saver()
		tf.initialize_all_variables().run()
		return siamese_net, train_step, saver


	def train_on_batch(data_x1, data_x2, data_y, cur_iter):
		_, loss = sess.run([train_step, siamese.loss], feed_dict={
                        siamese.x1: data_x1,
                        siamese.x2: data_x2,
                        siamese.y: data_y})
		if cur_iter % 10 == 0:
			print ('step %d: loss %.3f' % (step, loss))


if __name__ == "__main__":

	siamese_model = 

	# first load in pretrained embedding
	embeddings, word_id, id_word  = load_embedding("../embed/polyglot-zh.pkl")
	data_list = read_csv("../data/atec_nlp_sim_train.csv")
	train_data, test_data = prepare_train_test(data_list)

	siamese_net, train_step, saver = network_setup()

	sess = tf.session() # open a new session
	load_previous_model(sess) # ask if want to load a pre-trained model

	data_x1, data_x2, data_y = load_batches(train_data, batch_size=32) # load data into batches
	num_batches = len(data_x1) 
		


		for i in xrange(num_batches):
			train_on_batch(data_x1, data_x2, data_y)







