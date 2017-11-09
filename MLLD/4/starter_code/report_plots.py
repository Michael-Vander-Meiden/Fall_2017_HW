import numpy as np
import pdb
import matplotlib.pyplot as plt

epoch10 = np.load('epoch_times_10.npy')
epoch15 = np.load('epoch_times_15.npy')
epoch20 = np.load('epoch_times_20.npy')
epoch16 = np.load('epoch_times_16.npy')
epoch32 = np.load('epoch_times_32.npy')
epoch64 = np.load('epoch_times_64.npy')
epochs = np.arange(15)

plt.plot(epochs,epoch10, 'r')
plt.plot(epochs,epoch15, 'g')
plt.plot(epochs,epoch20, 'b')
plt.show()

