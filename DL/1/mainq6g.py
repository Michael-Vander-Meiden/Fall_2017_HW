import nn
import helper
import dataio
import numpy as np
import pdb
import matplotlib.pyplot as plt

training_data = dataio.training_from_doc()

validation_data = dataio.test_from_doc()
learning_rates = [.1,.01,.2,.5]

a = nn.FCN(architecture=[784,10,20,10], eta = 0.03, batch_size = 1, n_epochs=500, rand_seed = 40, momentum = 0.9)
a.fit(training_data,validation_data)

helper.plot_ce('q6g_ce',a)
helper.plot_ac('q6g_ac',a)


np.save('gbest_weights',a.weights)
