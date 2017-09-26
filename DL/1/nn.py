import random as rd
import numpy as np
import helper
import copy
import random
import pdb

#using a similar architecture to SKLearn's models

class FCN(object):
    def __init__(self, architecture=[784,100,10], eta = 0.1, batch_size = 1, dropout = 0, n_epochs=200, rand_seed = 1):
        

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eta = eta

        self.architecture = architecture
        self.num_layers = len(architecture)

        #initialize weights and biases from normal distribution

        #TODO change this into helper function
        #TODO check if we want random along a distribution

        #initializing weights and biases for all the layers as np array

        #this is a list of arrays of all the weights in an y by x shape for each layer
        self.weights = [np.array([0])] + [np.random.randn(y,x) for y,x in zip(architecture[1:], architecture[:-1])]

        #this is a list of 1d arrays, which have one bias per neuron
        self.biases = [np.random.randn(y,1) for y in architecture]

        #this creates a list of arrays of 0, for the inputs to each neuron
        self._inps = [np.zeros(bias.shape) for bias in self.biases]

        #this creates a list of arrays of 0, for the activations of each neuron. Same shape as inps.
        self._outs = copy.deepcopy(self._inps)

    def fit(self, training_data, validation_data=None):

        for epoch in range(self.n_epochs):
            #first we shuffle training data

            random.shuffle(training_data)

            #split into mini batches of the specified size
            mini_batches = [training_data[k:k + self.batch_size] for k in
                            range(0, len(training_data), self.batch_size)]

            for mini_batch in mini_batches:
                #create matrices to store our nablas
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]
                #perform forward and back prop on all examples in mini batch
                for x, y in mini_batch:
                    self._forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self._back_prop(x, y)
                    #calculate the new nablas 
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]


                #calculate new weights based on old weights, learn rate, and nablas
                self.weights = [w - (self.eta / self.batch_size) * dw for w, dw in zip(self.weights, nabla_w)]
                #calculate biases
                self.biases = [b - (self.eta / self.batch_size) * db for b, db in zip(self.biases, nabla_b)]

            #print status messages and get validation data
            if validation_data:
                accuracy = self.validate(validation_data) / 100.0
                print("Epoch {0}, accuracy {1} %.".format(epoch + 1, accuracy))
            else:
                print("Processed epoch {0}.".format(epoch))
                

    def validate(self,validation_data):
        validation_results = [(self.predict(x) == y) for x, y in validation_data]
        return sum(result for result in validation_results)     

    def predict(self, x):
        self._forward_prop(x)
        return np.argmax(self._outs[-1])

    def _forward_prop(self, x):
        #set activations equal to the input array
        self._outs[0] = x
        #process each layer using our weights, biases and activation function
        for i in range(1, self.num_layers):
            #calculate each neuron's pre-activation values
            self._inps[i] = (self.weights[i].dot(self._outs[i - 1]) + self.biases[i])
            #activate these values to get that layers outputs
            self._outs[i] = helper.sigmoid(self._inps[i])


#TODO look up cross-entropy and impliment backprop for it
    def _back_prop(self,x,y):
        #create a list of nabla arrays for each layer in the same shape of weights and biases
        delt_nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        delt_nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        #TODO change this to softmax
        error = (self._outs[-1] - y) * helper.dSigmoid(self._inps[-1])
        delt_nabla_b[-1] = error
        delt_nabla_w[-1] = error.dot(self._outs[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(self.weights[l + 1].transpose().dot(error),helper.dSigmoid(self._inps[l]))
            delt_nabla_b[l] = error
            delt_nabla_w[l] = error.dot(self._outs[l - 1].transpose())

        return delt_nabla_b, delt_nabla_w



    def load(self, filename='model.npz'):
        """Prepare a neural network from a compressed binary containing weights
        and biases arrays. Size of layers are derived from dimensions of
        numpy arrays.
        Parameters
        ----------
        filename : str, optional
            Name of the ``.npz`` compressed binary in models directory.
        """
        npz_members = np.load(os.path.join(os.curdir, 'models', filename))

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])

        # Bias vectors of each layer has same length as the number of neurons
        # in that layer. So we can build `sizes` through biases vectors.
        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

        # These are declared as per desired shape.
        self._inps = [np.zeros(bias.shape) for bias in self.biases]
        self._outs = [np.zeros(bias.shape) for bias in self.biases]

        # Other hyperparameters are set as specified in model. These were cast
        # to numpy arrays for saving in the compressed binary.
        self.batch_size = int(npz_members['batch_size'])
        self.n_epochs = int(npz_members['n_epochs'])
        self.eta = float(npz_members['eta'])

    def save(self, filename='model.npz'):
        """Save weights, biases and hyperparameters of neural network to a
        compressed binary. This ``.npz`` binary is saved in 'models' directory.
        Parameters
        ----------
        filename : str, optional
            Name of the ``.npz`` compressed binary in to be saved.
        """
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            weights=self.weights,
            biases=self.biases,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            eta=self.eta
)