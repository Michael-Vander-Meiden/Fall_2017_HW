import nn
import helper
import dataio
import numpy as np
import pdb
import matplotlib.pyplot as plt

training_data = dataio.training_from_doc()

validation_data = dataio.validation_from_doc()
test_data = dataio.test_from_doc()
learning_rates = [.1,.01,.2,.5]

a = nn.FCN(architecture=[784,100,10], eta = 0.5, batch_size = 1, n_epochs=18, rand_seed = 4, momentum = 0.0)
a.fit(training_data,test_data)

helper.plot_ce('q6f_ce',a)
helper.plot_ac('q6f_ac',a)


np.save('fbest_weights',a.weights)
