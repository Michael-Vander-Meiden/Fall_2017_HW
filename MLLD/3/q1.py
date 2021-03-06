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
    splits = line.split('\t')
    labels = tokenizeDoc(splits[1])
    words = tokenizeDoc(splits[2])
    return labels, words

def make_empty_list(n):
	num_labels = 5
	l = [0]*n
	for i in range(n):
		l[i] = [0]*5
	return l

def build_label_list(labels,cur_label_count,label_dictl,llist):
	a = []
	for label in labels:
		if label not in label_dict:
			label_dict[label] = cur_label_count
			cur_label_count += 1
			llist.append(label)
		a.append(label_dict[label])
	return cur_label_count, label_dict, llist

def labels_to_index(labels,label_dict):
	onehot = [0]*5
	for label in labels:
		index = int(label_dict[label])
		onehot[index] = 1
	return onehot

def dot_product (keys,B,x):
	a = [0]*5
	#print len(keys)
	for key in keys:
		for yi, j in enumerate(B[key]):
			value = x[key]*j
			a[yi]=a[yi]+value
	return a

def init_x(n):
	l = [0]*n
	return l


n = int(sys.argv[1])
init_l = float(sys.argv[2])
mu = float(sys.argv[3])
max_iter = sys.argv[4]
train_size = int(sys.argv[5])
test_file = sys.argv[6]
label_dict = dict()
cur_label_count = 0
llist = []
stopwords = ['a','the','of','be','to','and','in','that','have','it','for','not','on']

k = 0.0
A = make_empty_list(n)
B = make_empty_list(n)
LCLl = [0]*5
LCLavg = [0]*5
old_epoch = 0
for i,curdoc in enumerate(sys.stdin):
	#create epochs
	epoch = i/train_size
	if epoch != old_epoch:
		string = str(old_epoch)+","
		for ite,item in enumerate(LCLl):
			LCLavg[ite] = item/float(train_size)
		string = string + str(sum(LCLavg))
		print string 
		LCLavg = [0]*5
		LCLl = [0]*5
	old_epoch = epoch

	lam = init_l/((1+epoch)**2)
	x = dict()
	labels,words = split_line(curdoc)

	for word in stopwords:
		try:
			words.remove(word)
		except ValueError:
			pass
	if cur_label_count < 5:
		cur_label_count,label_dict,llist = build_label_list(labels,cur_label_count,label_dict,llist)

	l_index = labels_to_index(labels,label_dict)
	k = k+1
	keys = []
	for word in words:
		key = col_index(word,n)
		x[key]= x.get(key,0.0)+1.0
		if key not in keys:
			keys.append(key)
	dot = dot_product(keys,B,x)
	for yi, label_i in enumerate(l_index):
		p = sigmoid(dot[yi])
		if label_i == 1:
			LCLl[yi] =LCLl[yi]+ math.log(p)
		if label_i == 0:
			LCLl[yi] = LCLl[yi]+math.log(1-p)

		if (label_i-p)>0.5 or i%2 == 0:
			val1 = lam*(label_i-p)
			val2 = 1.0-2.0*lam*mu
			for key in keys:
				if abs(B[key][yi]) > .7:
					B[key][yi]=B[key][yi]*(val2)**(k-A[key][yi])
				B[key][yi] = B[key][yi] + val1
				A[key][yi] = k

#Test time!

with open(test_file) as f:
	correct = 0.0
	total = 0.0
	for i,curdoc in enumerate(f):
		ps = []
		final_string = ''
		x = [0]*n
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
		l_index = labels_to_index(labels,label_dict)
			
		for yi, label in enumerate(llist):

			p = math.exp(dot[yi])/(1+math.exp(dot[yi]))
			ps.append(p)
			final_string = final_string+label+'\t'+str(p)+','
		#print(final_string[:-1])
		if llist[ps.index(max(ps))] in labels:
			correct +=1
		total+=1

		




