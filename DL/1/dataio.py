import sys
import re
import numpy as np
import pdb
import copy

#Goal of this function is to take the name of a data file and output it in a useful format
#Our useful format will be tuples (X,Y) where X is the data and Y is the list


def data_from_doc(docname = 'digitstrain.txt'):
	

	lines = open(docname).readlines()
	holder = [(np.fromstring(line.rstrip()[:-2],dtype = float,sep=','),int(line.rstrip()[-1:]))for line in lines]
	data = copy.deepcopy(holder)

	for i,item in enumerate(holder):
		data[i] = (np.expand_dims(item[0], axis=1),onehot(item[1]))

	return data

def onehot(label):
	#this function will turn the label into a numpy array
	label_vec = np.zeros((10,1))
	label_vec[label] = 1
	return label_vec
