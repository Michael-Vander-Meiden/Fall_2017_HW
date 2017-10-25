import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
import pdb

weights = numpy.load('weightsq5e.npy')[1]

images = []
rows = 10
cols = 10
fig = plt.figure()
gs = gridspec.GridSpec(rows,cols,wspace=0.01)
ax = [plt.subplot(gs[i]) for i in range(rows*cols)]

pdb.set_trace()
for i in range(weights.shape[0]):
	im = weights[i,:].reshape((28,28))
	ax[i].imshow(im)
	ax[i].axis('off')

plt.savefig('q5e.png')
plt.show()

