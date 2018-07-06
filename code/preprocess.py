# this file is used for data precessing of the nlp task of Ma Yi Jin Fu challenge

import csv
import numpy as np

DEFAULT_WORD = '駧'
DEFAULT_SENT_SIZE = 25

def read_csv(file_path):
	'''
	Function: read in csv file, and store in the format of list of tuples
	'''

	data_list = []
	with open(file_path) as file:
		line = file.readline().strip()
		while line:
			item = line.split('\t') # split by tab, then in the format (number, sentence 1, sentence 2, label)
			
			sent1, mask1 = add_mask_and_padding(item[1])
			sent2, mask2 = add_mask_and_padding(item[2])

			data_list.append((sent1,
							  sent2,
							  int(item[3]),
							  mask1,
							  mask2))
			line = file.readline().strip()
	return data_list

def add_mask_and_padding(sent):
	mask1 = list(np.zeros(DEFAULT_SENT_SIZE))
	if len(sent) > DEFAULT_SENT_SIZE:
		mask1[-1] = 1
		new_sent = sent[:DEFAULT_SENT_SIZE]
	else:
		mask1[len(sent) - 1] = 1
		num_pads = DEFAULT_SENT_SIZE - len(sent)
		sent_list = list(sent)
		for i in range(num_pads):
			sent_list.append(DEFAULT_WORD)
		new_sent = ''.join(sent_list)
	return new_sent, mask1

def prepare_train_test(data_list):
	'''
	split data into training and testing set
	'''
	np.random.shuffle(data_list)
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
	mask_x1 = []
	mask_x2 = []


	for i in range(int(len(data_list) / batch_size)):
		start_ind = i * batch_size
		end_ind = start_ind + batch_size
		cur_data_x1 = []
		cur_data_x2 = []
		cur_data_y  = []
		cur_mask_x1 = []
		cur_mask_x2 = []
		for j in range(start_ind, end_ind):
			cur_data_x1.append(data_list[j][0]) # first sentence
			cur_data_x2.append(data_list[j][1]) # second sentence
			cur_data_y.append(data_list[j][2]) # label
			cur_mask_x1.append(data_list[j][3]) # first sentence mask
			cur_mask_x2.append(data_list[j][4]) # first sentence mask

		data_x1.append(cur_data_x1)
		data_x2.append(cur_data_x2)
		data_y.append(cur_data_y)
		mask_x1.append(cur_mask_x1)
		mask_x2.append(cur_mask_x2)


	return data_x1, data_x2, data_y, mask_x1, mask_x2









