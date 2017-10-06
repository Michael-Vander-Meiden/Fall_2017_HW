import matplotlib.pyplot as plt
import random
N = 30
G = 4

def spam_response(rule,k,G):
	if rule == 'mult':
		total_reward=0
		for i in range(k):
			cur_reward = 1
			for j in range(G):
				answer = random.randint(0,1)
				cur_reward = cur_reward*answer
			total_reward+= cur_reward
		return total_reward
	else:
		total_reward = 0
		for i in range(k):
			cur_reward = 0
			for j in range(G):
				answer = random.randint(0,1)
				cur_reward += float(answer)/4.0
			total_reward+=cur_reward
		return total_reward

def skip_response(rule,k,G):
	T = 0.6
	if rule == 'mult':
		total_reward=0
		for i in range(k):
			cur_reward = 1.0
			for j in range(G):
				cur_reward = cur_reward*T
			total_reward+= cur_reward
		return total_reward
	else:
		total_reward = 0
		for i in range(k):
			cur_reward = 0
			for j in range(G):
				cur_reward += (1.0*T)/4.0
			total_reward+=cur_reward
		return total_reward

def smart_response(rule,k,G):
	d = 0.9
	if rule == 'mult':
		total_reward=0
		for i in range(k):
			cur_reward = 1.0
			for j in range(G):
				if random.random()>0.9:
					cur_reward = 0
			total_reward+= cur_reward
		return total_reward
	else:
		total_reward = 0
		for i in range(k):
			cur_reward = 0
			for j in range(G):
				if random.random()<0.9:
					cur_reward += (1.0)/4.0
			total_reward+=cur_reward
		return total_reward	


X1 = []
X2 = []
Y = []
#spam 
'''
for i in range(1000):
	mult = spam_response('mult',i,G)
	add = spam_response('add',i,G)
	X1.append(mult)
	X2.append(add)
	Y.append(i)
	'''
for i in range(1000):
	mult = smart_response('mult',i,G)
	add = smart_response('add',i,G)
	X1.append(mult)
	X2.append(add)
	Y.append(i)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(Y,X1)
ax1.plot(Y,X2)
plt.show()
