import sys
import re
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
	l = [None]*n
	for i in range(n):
		l[i] = [None]*5
	return l

def labels_to_index(labels,cur_label_count,label_dict):
	a = []
	for label in labels:
		if label not in label_dict:
			label_dict[label] = cur_label_count
			cur_label_count += 1
		a.append(label_dict[label])
	return a, cur_label_count, label_dict


n = int(sys.argv[1])
lam = sys.argv[2]
mu = sys.argv[3]
max_iter = sys.argv[4]
train_size = sys.argv[5]
test_file = sys.argv[6]
label_dict = {}
cur_label_count = 0


sys.argv

k = 0.0
#TODO make A nd B list of lists instead of dicts
A = make_empty_list(n)
B = make_empty_list(n)
#TODO figure out how to turn label into index


for i,curdoc in enumerate(sys.stdin):
	labels,words = split_line(curdoc)
	l_index, cur_label_count,label_dict = labels_to_index(labels,cur_label_count,label_dict)
	k = k+1
	for label_i in label_index:
		i=1
		for word in words:
			key = col_index(word,n)
			if B[key][i] == None:
				B[key][i]=0.0
				A[key][i]=0.0
'''
		B[key]=B[key]*(1.0-2.0*lam*mu)^(k-A[key])
		B[key] = B[key] + lam*()
'''
