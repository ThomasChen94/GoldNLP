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
			item = line.split('\t') # split by tab, then in the format (number, sentence 1, sentence 2, label)

			data_list.append((item[1],
							  item[2],
							  item[3]))
			line = file.readline()

	return data_list


