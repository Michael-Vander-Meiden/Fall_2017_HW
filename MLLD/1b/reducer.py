#!/usr/bin/env python
import re 
import sys
import pdb
import time
import math

def tokenizeDoc(cur_doc):
	return re.findall('\\w+', cur_doc)

def tab_split(line):
	splits = re.split(r'\t+', line)
	key = splits[0]
	value = float(splits[1])
	return key, value

def split_doc(cur_doc):
	splits = re.split(r'\t+', cur_doc)
	labels = splits[0]
	words = splits[1]
	return labels, words


#initiate dict
SNB = {}

#read input from training
for pair in sys.stdin:
	key, value = tab_split(pair)
	SNB[key]=SNB.get(key,0.0)+float(value)

for i in SNB:
	print(i+"\t"+str(SNB[i]))