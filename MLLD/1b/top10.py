#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
SNB = dict()
top10s = []
labels = []

#read input from training
for pair in sys.stdin:
	key, value = tab_split(pair)
	#Get unique keys for top10s
	label = key.split(',')[0]
	if label not in labels and len(label) > 5:
		labels.append(label)
	SNB[key]=value


for label in labels:
	print label
	curdict = dict()
	for i in SNB:
		thing = i.split(',')[0]
		if (thing ==label) and (len(i.split('=')) == 3):
			curdict[i] = SNB[i]
	curtop10 = sorted(curdict, key=curdict.get, reverse=True)[1:11]
	print(curtop10)
	top10s.append(curtop10)

with open("top10.txt", "w") as text_file:
	for mylist in top10s:
		for key in mylist:
			myclass = key.split('=')[1][:-2]
			print(key)
			myword = key.split('=')[2]
			count = SNB[key]
			text_file.write(myclass+"\t"+myword+"\t"+str(int(count))+"\n")
text_file.close()
