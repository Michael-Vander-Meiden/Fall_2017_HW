import nn
import helper
import dataio
import numpy as np
import pdb
import matplotlib.pyplot as plt

training_data = dataio.training_from_doc()

validation_data = dataio.validation_from_doc()
learning_rates = [.1,.01,.2,.5]

a = nn.FCN(rand_seed=5, eta=.01, momentum=0.5, architecture=[784,100,10],n_epochs=200)
b = nn.FCN(rand_seed=5, eta=.01, momentum=0.5, architecture=[784,20,10],n_epochs=200)
c = nn.FCN(rand_seed=5, eta=.01, momentum=0.5, architecture=[784,200,10],n_epochs=200)
d = nn.FCN(rand_seed=5, eta=.01, momentum=0.5, architecture=[784,500,10],n_epochs=200)


best_results = 0
best_network = 5
for net,file in zip([a,b,c,d],['100hu','20hu','200hu','500hu']):
	net.fit(training_data,validation_data)
	helper.plot_ce(file+'ce',net)
	helper.plot_ac(file+'ac',net)
