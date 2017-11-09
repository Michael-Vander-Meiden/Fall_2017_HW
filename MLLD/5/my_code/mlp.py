"""
Multilayer Perceptron for character level entity classification
"""
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *
import copy
import pdb
import time


np.random.seed(234)

class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        self.my_xman = self._build(layer_sizes) # DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self,layer_sizes):
        x = XMan()
        #TODO define your model here

        #layer sizes is a tuple with the size of each layer
        i_size = layer_sizes[0]
        o_size = layer_sizes[-1]

        #TODO declare inputs and labels
        x.out0 = f.input(name="out0", default=np.random.rand(1,i_size))
        def_y = np.zeros((1,o_size))
        def_y[0,np.random.choice(o_size)]=1
        x.y = f.input(name="y", default=def_y)
        
        #TODO declare bias and weights for all layers using loop
        
        for i in range(len(layer_sizes)-1):
            din = layer_sizes[i]
            dout = layer_sizes[i+1]
            a = (6.0/(din+dout))**0.5
            w_name = 'w{0}'.format(i)
            b_name = 'b{0}'.format(i)
            out_name = 'out{0}'.format(i+1)
            input_name = 'out{0}'.format(i)
            setattr(x, w_name, f.param(name=w_name, default=np.random.uniform(-a,a,(din,dout))))
            setattr(x, b_name, f.param(name=b_name, default=np.random.uniform(-0.1,0.1,dout)))
            setattr(x, out_name, f.relu(f.mul(getattr(x,input_name),getattr(x,w_name))+getattr(x,b_name)))
            x.outputs = f.softMax(getattr(x,out_name))
            x.loss = f.mean(f.crossEnt(x.outputs,x.y))
        return x.setup()

def check_gradient(key, value_dict, ad, wengert_list, gradients, epsilon=1e-8):
    for index, value in np.ndenumerate(value_dict[key]):
        value_dict[key][index] += epsilon
        big = ad.eval(wengert_list, value_dict)['loss']
        value_dict[key][index] -= (2 * epsilon)
        small = ad.eval(wengert_list, value_dict)['loss']
        value_dict[key][index] += epsilon
        assert abs(value - value_dict[key][index]) < 0.0000001
        diff = abs((big-small) / (2*epsilon) - gradients[key][index])
        print 'manual grad:', (big-small) / (2*epsilon)
        print 'auto grad:', gradients[key][index]
        print diff
        #if diff > 0.001:
        #    return False
    exit()
    return True



def main(params,check_grad=True):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']
    train_loss_file = params['train_loss_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)

    # build
    print "building mlp..."
    mlp = MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
    #TODO CHECK GRADIENTS HERE

    print "done"

    # train
    print "training..."
    # get default data and params
    value_dict = mlp.my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    best_vloss = 1000.
    #does this stuff go here?
    ad = Autograd(mlp.my_xman)
    wlist = mlp.my_xman.operationSequence(mlp.my_xman.loss)


    epoch_times = np.ndarray([0])
    for i in range(epochs):
        start = time.time()
        print "Epoch {}".format(i)
        total_train_loss = 0.
        train_counter = 0.
        for (idxs,e,l) in mb_train:
            #TODO prepare the input and do a fwd-bckwd pass over it and update the weights
            e = np.array(e)
            l = np.array(l)
            e = e.reshape(e.shape[0],-1)
            value_dict['out0'] = e
            value_dict['y'] = l
            value_dict = ad.eval(wlist,value_dict)
            grads = ad.bprop(wlist,value_dict,loss=np.float_(1.))

            for key in grads:
                if mlp.my_xman.isParam(key):
                    value_dict[key] -= lr*grads[key]

            #save the train loss
            train_counter += 1
            total_train_loss+=value_dict['loss']
            train_loss = np.append(train_loss, value_dict['loss'])
        average_batch_loss = total_train_loss/train_counter
        print "training loss is: {}".format(average_batch_loss)
                                   
        # validate
        total_val_loss = 0.
        val_counter = 0.
        for (idxs,e,l) in mb_valid:
            #TODO prepare the input and do a fwd pass over it to compute the loss
            e = np.array(e)
            l = np.array(l)
            e = e.reshape(e.shape[0],-1)

            value_dict['y'] = l   
            value_dict['out0'] = e         
            value_dict = ad.eval(wlist,value_dict)
            val_counter+=1.
            total_val_loss+=value_dict['loss']
        average_batch_loss = total_val_loss/val_counter
        print "validation loss is:{}".format(average_batch_loss)
        #TODO compare current validation loss to minimum validation loss
        if average_batch_loss < best_vloss:
            print"Storing new best dict"
            best_vloss=average_batch_loss
            best_dict = copy.deepcopy(value_dict)
        # and store params if needed
        epoch_times = np.append(epoch_times,(time.time()-start))
    print "done"
    #write out the train loss
    np.save(train_loss_file, train_loss)    
    
    test_counter = 0.
    total_test_loss = 0.
    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        test_counter += 1
        e = np.array(e)
        l = np.array(l)
        e = e.reshape(e.shape[0],-1)
        best_dict['out0'] = e
        best_dict['y'] = l            
        best_dict = ad.eval(wlist,best_dict)
        total_test_loss += best_dict['loss']
    print total_test_loss/test_counter
    #TODO save probabilities on test set
    # ensure that these are in the same order as the test input
    np.save(output_file, best_dict['outputs'])
    np.save('epoch_times_20', epoch_times)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=20)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='smaller')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--train_loss_file', dest='train_loss_file', type=str, default='train_loss')
    params = vars(parser.parse_args())
    main(params)
