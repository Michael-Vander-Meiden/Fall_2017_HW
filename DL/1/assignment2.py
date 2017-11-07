import nn
import helper
import dataio
import numpy as np
import pdb
import matplotlib.pyplot as plt

training_data = dataio.training_from_doc()
validation_data = dataio.validation_from_doc()
learning_rates = [.1,.01,.2,.5]
autoencoder = np.load('weightsq5e.npy')[1]
denoise = np.load('weightsq5f.npy')[1]
denoise.tolist()
rbm = np.load('a2weights.npy')
rbm.tolist()
autoencoder.tolist()


c = nn.FCN(rand_seed=1,pretrain=denoise)

a = nn.FCN(rand_seed=1,pretrain=autoencoder)
b = nn.FCN(rand_seed=1,pretrain=rbm)

#pdb.set_trace()
a.fit(training_data, validation_data)
b.fit(training_data, validation_data)
c.fit(training_data, validation_data)
'''
plt.plot(a.epoch_nums,vce, 'r--', tce, 'b--')
plt.savefig('a2q5pre.png')
plt.show()
'''
plt.plot(a.epoch_nums,b.v_ac_array, 'b--')
plt.plot(a.epoch_nums,a.v_ac_array, 'r--')
plt.plot(a.epoch_nums,c.v_ac_array, 'g--')
plt.savefig('a2q5f.png')
plt.show()
pdb.set_trace()
