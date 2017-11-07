The Autoencoder is a python class in NN. The parameters can be seen in the class's init function. 

To train the network, simply use the main function, and create a class of autoencoder with the parameters you'd like. To make the autoencoder become a denoising autoencoder, set dropout=True during initialization.

Then, using a.fit(test_data,train_data) you can train the network on your dataset.

Below I saved some plots for the desired loss metrics.

If you would like to view the weights, you can simply call the class variable "weights"

The helper file and dataio files assist with data and some functions to make the rest of the code less cluttered.

For the RBM, the rbm is a class in "models.py". The init shows some of the adjustable parameters. 
"main.py" shows an example of it's use, along with plotting the figures of cross entropy at the end.

There is also a "dataio" helper function.
