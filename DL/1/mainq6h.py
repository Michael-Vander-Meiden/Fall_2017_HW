import nn
import helper
import dataio
import numpy as np
import pdb
import matplotlib.pyplot as plt

training_data = dataio.training_from_doc()

validation_data = dataio.validation_from_doc()
learning_rates = [.1,.01,.2,.5]

a = nn.FCN(architecture=[784,100,100,10], eta = 0.5, batch_size = 32, n_epochs=250, rand_seed = 41, momentum = .5)
a.fit(training_data,validation_data)

helper.plot_ce('h6g_ce',a)
helper.plot_ac('h6g_ac',a)


np.save('hbest_weights',a.weights)
