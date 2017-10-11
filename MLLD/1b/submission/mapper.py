#!/usr/bin/env python
import re 
import sys
import pdb
import time

def tokenizeDoc(cur_doc):
	return re.findall('\\w+', cur_doc)

def split_doc(cur_doc):
	splits = re.split(r'\t+', cur_doc)
	labels = splits[1]
	words = splits[2]
	return labels, words

def nbCount(labels,words,nbdict,vocab):
	for label in labels:
		for word in words:
			vocab[word]="yes"
		#Increase this labels number of instances
		l_key = 'Y='+label
		nbdict[l_key]=nbdict.get(l_key,0.0)+1.0
		#increase total number of lable instances
		nbdict['Y=*']=nbdict.get('Y=*',0.0)+1.0

		for word in words:
			#increase instance of word in class
			w_key = "Y="+label+',W='+word
			nbdict[w_key]=nbdict.get(w_key,0.0)+1.0
			#increase total number of words in class
			any_w_key = "Y="+label+',W=*'
			nbdict[any_w_key]=nbdict.get(any_w_key,0.0)+1.0
'''
def outLabelString(label)
	mystring = "Y="+label
	otherString = "Y=ANY"
	output1 = mystring + "\t" + "1"

def outWordString(label,word)
#data = sys.stdin.read()
SNB = dict()
vocab = dict()
total_ECAT=0

for curdoc in sys.stdin:
	labels,words = split_doc(curdoc)

	labels = tokenizeDoc(labels)
	words = tokenizeDoc(words)

	for label
	
'''
'''
for curdoc in sys.stdin:
	labels,words = split_doc(curdoc)

	labels = tokenizeDoc(labels)
	

	words = tokenizeDoc(words)
	SNB = dict()
	vocab = dict()
	nbCount(labels,words,SNB,vocab)
	SNB["V"] = len(vocab)
	for i in SNB:
		print(i+"\t"+str(SNB[i]))

'''


'''
for i,curdoc in enumerate(data.splitlines()):
	labels,words = split_doc(curdoc)

	labels = tokenizeDoc(labels)
	words = tokenizeDoc(words)
	SNB = dict()
	vocab = dict()
	nbCount(labels,words,SNB,vocab)
	SNB["V"] = len(vocab)
	for i in SNB:
		print(i+"\t"+str(SNB[i]))

'''