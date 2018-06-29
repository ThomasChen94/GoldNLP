from preprocess import read_csv

import pickle
import numpy







if __name__ == "__main__":
	
	words, embeddings = pickle.load(open('/home/polyglot/en/words_embeddings_32.pkl', 'rb'))
	print("Emebddings shape is {}".format(embeddings.shape))