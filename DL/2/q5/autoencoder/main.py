import nn
import helper
import dataio
import numpy as np
import pdb
import matplotlib.pyplot as plt

training_data = dataio.training_from_doc()
validation_data = dataio.validation_from_doc()
learning_rates = [.1,.01,.2,.5]

a = nn.FCN(rand_seed=1, dropout=True)

#pdb.set_trace()
a.fit(training_data, validation_data)

np.save('weightsq5f',a.weights)

#pdb.set_trace()

vce=a.v_ce_array
tce=a.t_ce_array

plt.plot(a.epoch_nums,vce, 'r--')
plt.plot(a.epoch_nums,tce, 'b--')
plt.savefig('q5f.png')
plt.show()
