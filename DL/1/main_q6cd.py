import nn
import helper
import dataio
import numpy as np
import pdb
import matplotlib.pyplot as plt

training_data = dataio.training_from_doc()

validation_data = dataio.validation_from_doc()
learning_rates = [.1,.01,.2,.5]

a = nn.FCN(rand_seed=1,eta=0.1 )
b = nn.FCN(rand_seed=2, eta=.01)
c = nn.FCN(rand_seed=3, eta=.2)
d = nn.FCN(rand_seed=4, eta=.5)
e = nn.FCN(rand_seed=5, eta=.1, momentum=0.5)
f = nn.FCN(rand_seed=5, eta=.1, momentum=0.9)


best_results = 0
best_network = 5
for net,file in zip([a,b,c,d,e,f],['eta01','eta001','eta02','eta05','m5','m9']):
	net.fit(training_data,validation_data)
	helper.plot_ce(file+'ce',net)
	helper.plot_ac(file+'ac',net)
	if net.v_ac_array[-1] > best_results:
		best_network = net

np.save('best_weights',best_network.weights)
