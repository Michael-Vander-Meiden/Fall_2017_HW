import numpy as np


#Below are activation functions and their derivatives

def sigmoid(inp):
	outp = 1.0 / (1.0 + np.exp(-inp))
	return outp

def dSigmoid(inp):
	outp = sigmoid(inp) * (1 - sigmoid(inp))
	return outp

def relu(inp):
	outp = np.maximum(z, 0)
	return outp

def dRelu(inp):
	outp = float(z>0)
	return outp

#softmax function for final layer

def softmax(inp):
	outp = (np.exp(inp) / np.sum(np.exp(inp)))
	return outp

def dSoftmax(inp):
	outp = softmax(inp)*(1-softmax(inp))
