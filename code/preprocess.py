# this file is used for data precessing of the nlp task of Ma Yi Jin Fu challenge

import csv

def read_csv(file_path):
	'''
	Function: read in csv file, and store in the format of list of tuples
	'''

	data_list = []
	with open(file_path) as file:
		line = file.readline()
		while line:
			item = line.decode('utf8').split('\t') # split by tab, then in the format (number, sentence 1, sentence 2, label)

			data_list.append((item[1],
							  item[2],
							  item[3]))
			line = file.readline()

	return data_list

def prepare_train_test(data_list):
	'''
	split data into training and testing set
	'''
	data_list = shuffle(data_list)
	split_ind = int(0.8 * len(data_list)) # split training and testing by 8:2
	train_data = data_list[:split_ind] 
	test_data = data_list[split_ind:len(data_list)]
	return train_data, test_data

def load_batches(data_list, batch_size):
	'''
	load data into batches
	'''

	data_x1 = []
	data_x2 = []
	data_y = []

	for i in xrange(len(data_list) / batch_size):
		start_ind = i * batch_size
		end_ind = start_ind + batch_size

		data_x1.append(data_list[start_ind : end_ind][0]) # first sentence
		data_x2.append(data_list[start_ind : end_ind][1]) # second sentence
		data_y.append(data_list[start_ind : end_ind][2]) # label


	return data_x1, data_x2, data_y









