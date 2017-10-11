import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec

weights = numpy.load('gbest_weights.npy')
images = []
rows = 3
cols = 4
fig = plt.figure()
gs = gridspec.GridSpec(rows,cols,wspace=0.01)
ax = [plt.subplot(gs[i]) for i in range(rows*cols)]

for i,weight in enumerate(weights[1]):
	im = weight.reshape((28,28))
	ax[i].imshow(im)
	ax[i].axis('off')
	
plt.show()

