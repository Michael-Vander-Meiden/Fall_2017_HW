import pdb
import numpy as np
import random


class RBM(object):
	def __init__(self, n_visible, n_hidden, alpha=0.01, W=None):
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		self.alpha = alpha
		b = np.ones((self.n_hidden,))
		c = np.ones((self.n_visible,))
		
		if W is None:
			W = np.random.normal(loc=0., scale=0.1, size=(self.n_hidden,self.n_visible))

		self.b = b
		self.c = c
		self.W = W

		self.avg_train_loss = list()
		self.avg_val_loss = list()
		self.epoch_nums = list()

	def _v_to_h(self,v):
		return sigmoid(np.dot(self.W,v)+self.b)

	def _h_to_v(self,h):
		return sigmoid(np.dot(self.W.transpose(),h)+self.c)

	def _prob_to_binary(self,P):
		randoms = np.random.uniform(low=0.0,high=1.0,size=P.shape)
		return (randoms<P)*1.0

	def train(self,X,X_val,k,num_epochs):
		for epoch in range(num_epochs):
			print("Epoch {0}:".format(epoch+1))
			np.random.shuffle(X)
			loss = 0.0
			val_loss = 0.0
			for i in range(X.shape[0]):
				x = X[i,:]
				loss += self._get_loss(x)
				cur_v = x
				cur_v_k = self._gibbs_chain(cur_v,k,True)
				self.backprop(cur_v,cur_v_k)

			for i in range(X_val.shape[0]):
				x = X_val[i,:]
				val_loss += self._get_loss(x)

			print loss/float(X.shape[0])
			print val_loss/float(X_val.shape[0])
			self.avg_train_loss.append(loss/float(X.shape[0]))
			self.avg_val_loss.append(val_loss/float(X_val.shape[0]))
			self.epoch_nums.append(epoch)



	def _gibbs_chain(self,v,k,sample):
		cur_v = v
		for i in range(k):
			cur_h_probs = self._v_to_h(cur_v)
			if sample:
				cur_h = self._prob_to_binary(cur_h_probs)
			else:
				cur_h = cur_h_probs

			cur_v_probs = self._h_to_v(cur_h)			

			if sample:
				cur_v = self._prob_to_binary(cur_v_probs)
			else:
				cur_v = cur_v_probs
		return cur_v

	def backprop(self,v,v_k):
		h = self._v_to_h(v)
		h_k = self._v_to_h(v_k)
		#make sure dimensions are good
		h = np.expand_dims(h,axis=1)
		h_k = np.expand_dims(h_k,axis=1)
		v = np.expand_dims(v,axis=1)
		v_k = np.expand_dims(v_k,axis=1)
		#pdb.set_trace()

		W_update = self.alpha*(np.dot(h,v.transpose()) - np.dot(h_k,v_k.transpose()))

		self.W = self.W + W_update
		self.b += (self.alpha*(h-h_k))[:,0]
		self.c += (self.alpha*(v-v_k))[:,0]

	def _get_loss(self,v):
		x = v
		y = self._gibbs_chain(x,1,False)
		#pdb.set_trace()
		loss = squared_error(x,y)
		return loss

def cross_entropy(x,y):
	loss = sum(-1.0*y*np.log(x) - (1.0-y)*np.log(1.0-x))
	return loss

def squared_error(x,y):
	loss = sum((x-y)**2)
	return loss
def sigmoid(inp):
	outp = 1.0 / (1.0 + np.exp(-inp))
	return outp

def dSigmoid(inp):
	outp = sigmoid(inp) * (1 - sigmoid(inp))
	return outp

def relu(inp):
	outp = np.maximum(inp, 0)
	return outp

def dRelu(inp):
	outp = float(inp>0)
	return outp