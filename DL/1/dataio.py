import sys
import re
import numpy as np
import pdb

#Goal of this function is to take the name of a data file and output it in a useful format
#Our useful format will be tuples (X,Y) where X is the data and Y is the list


def data_from_doc(docname = 'digitstrain.txt'):
	

	lines = open(docname).readlines()
	data = [(np.fromstring(line.rstrip()[:-2],dtype = float,sep=','),int(line.rstrip()[-1:]))for line in lines]
	return data


