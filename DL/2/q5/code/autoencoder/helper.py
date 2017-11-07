import numpy as np
import pdb
import matplotlib.pyplot as plt
#Below are activation functions and their derivatives

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

def softmax(inp):
	outp = (np.exp(inp) / np.sum(np.exp(inp)))
	return outp

def dSoftmax(inp):
	outp = softmax(inp)*(1-softmax(inp))

#Testing helper functions
def ce(x,y,mytype):
	if mytype == 'train':
		pdb.set_trace()

def plot_ce(filename, net):
	plt.clf()
	plt.plot(net.epoch_nums,net.v_ce_array, 'r--',net.t_ce_array, 'b--')
	plt.savefig(filename)

def plot_ac(filename, net):
	plt.clf()
	plt.plot(net.epoch_nums,net.v_ac_array, 'r--',net.t_ac_array, 'b--')
	plt.savefig(filename)	


