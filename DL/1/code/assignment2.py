import nn
import helper
import dataio
import numpy as np
import pdb
import matplotlib.pyplot as plt

training_data = dataio.training_from_doc()
validation_data = dataio.validation_from_doc()
learning_rates = [.1,.01,.2,.5]

a = nn.FCN(rand_seed=1)

#pdb.set_trace()
a.fit(training_data, validation_data,rbm)

plt.plot(a.epoch_nums,vce, 'r--', tce, 'b--')
plt.savefig('q6a.png')
plt.show()

plt.plot(a.epoch_nums,va, 'r--', ta, 'b--')
plt.savefig('q6b.png')
plt.show()
