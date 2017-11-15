import numpy as np


epochs = 100

obvious_error = 0
other_error = 0
my_error = 0
for i in range(epochs):

	x1 = np.random.uniform(low=-1,high=0)
	x2 = np.random.uniform(low=-2,high=0)
	x3 = np.random.uniform(low=0,high=1)
	x4 = np.random.uniform(low=0,high=3)
	x5 = np.random.uniform(low=0,high=1)
	x6 = np.random.uniform(low=-4,high=4)

	x = [x1,x2,x3,x4,x5,x6]

	y1 = np.random.normal(x1,1) 
	y2 = np.random.normal(x2,1) 
	y3 = np.random.normal(x3,1) 
	y4 = np.random.normal(x4,1) 
	y5 = np.random.normal(x5,1) 
	y6 = np.random.normal(x6,1) 

	y = [y1,y2,y3,y4,y5,y6]
	square_y = [b**2 for b in y]
	y_term = sum(square_y)
	myterm = sum(y)/len(y)

	#calculate obivous error
	curerr = 0
	for e,item in enumerate(x):
		curerr += (x[e]-y[e])**2

	obvious_error += curerr
	
	curerr = 0
	for e,item in enumerate(x):
		x_hat = (1.-(4./y_term))*y[e]
		curerr += (x[e]-x_hat)**2
	other_error += curerr

	curerr = 0
	for e,item in enumerate(x):
		curerr += (x[e]-0.5*(y[e]+myterm))**2
	
	my_error += curerr
print obvious_error
print other_error
print my_error




