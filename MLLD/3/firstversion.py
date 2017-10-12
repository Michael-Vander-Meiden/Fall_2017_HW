import sys
import re
import math
#this returns a non-unique ID. s is the string and n is the 
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

def labels_to_index(labels,cur_label_count,label_dict):
	a = []
	for label in labels:
		if label not in label_dict:
			label_dict[label] = cur_label_count
			cur_label_count += 1
		a.append(label_dict[label])

	onehot = [0]*5
	for label in a:
		onehot[label] = 1
	return onehot, cur_label_count, label_dict

def dot_product (x,B):
	sums = {}
	for i,item in enumerate(B):
		for yi, j in enumerate(item):
			value = float(x[i])*float(B[i][yi])

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


k = 0.0
A = make_empty_list(n)
B = make_empty_list(n)

for i,curdoc in enumerate(sys.stdin):
	#initialize x
	x = init_x(n)
	#create epochs
	epoch = i/train_size
	lam = init_l/((1+epoch)**2)
	labels,words = split_line(curdoc)
	l_index, cur_label_count,label_dict = labels_to_index(labels,cur_label_count,label_dict)
	k = k+1
	for word in words:
		key = col_index(word,n)
		x[key] = 1

	dot = dot_product(x,B)
	for yi, label_i in enumerate(l_index):

		p = math.exp(dot[yi])/(1+math.exp(dot[yi]))
		for word in words:
			key = col_index(word,n)
			B[key][yi]=B[key][yi]*(1.0-2.0*lam*mu)**(k-A[key][yi])
			B[key][yi] = B[key][yi] + lam*(label_i-p)*x[key]
			A[key][yi] = k

#Test time!
with open(test_file) as f:
	for i,curdoc in enumerate(f):
		ps = []
		final_string = ''
		#populate x
		labels,words = split_line(curdoc)
		x = init_x(n)
		for word in words:
			key = col_index(word,n)
			x[key] = 1
		dot = dot_product(x,B)
		l_index, cur_label_count,label_dict = labels_to_index(labels,cur_label_count,label_dict)
			
		for yi, label in enumerate(label_dict):

			p = math.exp(dot[yi])/(1+math.exp(dot[yi]))
			final_string = final_string+label+'\t'+str(p)+','

		print(final_string[:-1])
		




