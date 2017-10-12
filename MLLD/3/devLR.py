import sys
import re
import math
import time
from copy import copy
#this returns a non-unique ID. s is the string and n is the 
start_time = time.time()
def col_index(s,n):
	return hash(s)%n

def sigmoid(score):
	overflow = 20.0
	if score>overflow:
		score = overflow
	elif score < -overflow:
		score = -overflow
	exp = math.exp(score)
	return exp/(1+exp)

def tokenizeDoc(cur_doc):
    return re.findall('\\w+', cur_doc)

def split_line(line):
    splits = re.split(r'\t+', line)
    labels = tokenizeDoc(splits[1])
    words = tokenizeDoc(splits[2])
    return labels, words

def make_empty_list(n):
	num_labels = 5
	l = [0]*n
	for i in range(n):
		l[i] = [0]*5
	return l

def labels_to_index(labels,cur_label_count,label_dict,llist):
	a = []
	for label in labels:
		if label not in label_dict:
			label_dict[label] = cur_label_count
			cur_label_count += 1
			llist.append(label)
		a.append(label_dict[label])

	onehot = [0]*5
	for label in a:
		onehot[label] = 1
	return onehot, cur_label_count, label_dict, llist

def dot_product (keys,B,x):
	sums = {}
	for key in keys:
		for yi, j in enumerate(B[key]):
			value = x[key]*float(B[key][yi])
			sums[yi]= sums.get(yi,0.0)+value
	return sums

def init_x(n):
	l = [0]*n
	return l


n = int(sys.argv[1])
init_l = float(sys.argv[2])
mu = float(sys.argv[3])
max_iter = sys.argv[4]
train_size = int(sys.argv[5])
test_file = sys.argv[6]
label_dict = {}
cur_label_count = 0
llist = []

k = 0.0
A = make_empty_list(n)
B = make_empty_list(n)
x_base = init_x(n)
for i,curdoc in enumerate(sys.stdin):
	#create epochs
	epoch = i/train_size
	lam = init_l/((1+epoch)**2)
	x = copy(x_base)
	labels,words = split_line(curdoc)
	l_index, cur_label_count,label_dict,llist = labels_to_index(labels,cur_label_count,label_dict,llist)
	k = k+1
	keys = []
	for word in words:
		key = col_index(word,n)
		x[key]+=1
		if key not in keys:
			keys.append(key)
	dot = dot_product(keys,B,x)
	for yi, label_i in enumerate(l_index):
		p = sigmoid(dot[yi])
		if (label_i-p)>0.01 or i%2 == 0:
			for key in keys:
				if abs(B[key][yi]) > 0.8:
					B[key][yi]=B[key][yi]*(1.0-2.0*lam*mu)**(k-A[key][yi])
				B[key][yi] = B[key][yi] + lam*(label_i-p)*1.0
				A[key][yi] = k

#Test time!
stopwords = ['a','the','of','be','to','and','in','that','have','it','for','not','on']
with open(test_file) as f:
	correct = 0.0
	total = 0.0
	for i,curdoc in enumerate(f):
		ps = []
		final_string = ''
		x = init_x(n)
		#populate x
		labels,words = split_line(curdoc)
		keys = []
		
		for word in stopwords:
			try:
				words.remove(word)
			except ValueError:
				pass

		for word in words:
			key = col_index(word,n)
			x[key]+=1
			if key not in keys:
				keys.append(key)
		dot = dot_product(keys,B,x)
		l_index, cur_label_count,label_dict,llist = labels_to_index(labels,cur_label_count,label_dict,llist)
			
		for yi, label in enumerate(llist):

			p = math.exp(dot[yi])/(1+math.exp(dot[yi]))
			ps.append(p)
			final_string = final_string+label+'\t'+str(p)+','
		print(final_string[:-1])
		if llist[ps.index(max(ps))] in labels:
			correct +=1
		total+=1
print (correct/total)
print (time.time() - start_time)

		




