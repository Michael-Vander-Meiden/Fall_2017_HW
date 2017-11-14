"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *

np.random.seed(0)

class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size): 
        self.max_len = max_len
        self.in_size = in_size
        self.num_hid = num_hid
        self.out_size = out_size
        self.my_xman = self._build() #DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):
        x = XMan()
        #TODO: define your model here

        #example batch size for initialization
        batch_n = 5
    #We need to initialize all of the weights and biases, which we can do according to Glorot initialization

        #First step, find the scale of each initialization:    
        a_w1 = (6. / (self.in_size+self.num_hid))**0.5
        a_w2 = (6. / (self.out_size+self.num_hid))**0.5
        a_b = 0.1
        a_u = (6. / (self.num_hid*2)) ** 0.5

        #here we can code our default input and output
        #set each charachter x1,x2,x3,x4..
        y = np.zeros((batch_n, self.out_size))
        for i in range(batch_n):
            y[i, np.random.choice(self.out_size)] = 1
        x.y = f.input(name = "y", default = y)
        x.x = [f.input(name = 'x' + str(i+1), default = np.random.rand(1,self.in_size)) for i in range(self.max_len)]
        print ['x' + str(i+1) for i in range(self.max_len)]
        #defining variables using setattr
        def_W = np.random.uniform(low=-a_w1,high=a_w1,size=(self.in_size, self.num_hid))
        setattr(x,'Wi', f.param(name='Wi', default = def_W))
        setattr(x,'Wf', f.param(name='Wf', default = np.random.uniform(-a_w1,a_w1,size=(self.in_size,self.num_hid))))
        setattr(x,'Wo', f.param(name='Wo', default = np.random.uniform(-a_w1,a_w1,size=(self.in_size,self.num_hid))))
        setattr(x,'Wc', f.param(name='Wc', default = np.random.uniform(-a_w1,a_w1,size=(self.in_size,self.num_hid))))
        setattr(x,'Ui', f.param(name='Ui', default = np.random.uniform(-a_u,a_u,size=(self.num_hid,self.num_hid))))
        setattr(x,'Uf', f.param(name='Uf', default = np.random.uniform(-a_u,a_u,size=(self.num_hid,self.num_hid))))
        setattr(x,'Uo', f.param(name='Uo', default = np.random.uniform(-a_u,a_u,size=(self.num_hid,self.num_hid))))
        setattr(x,'Uc', f.param(name='Uc', default = np.random.uniform(-a_u,a_u,size=(self.num_hid,self.num_hid))))
        setattr(x,'bi', f.param(name='bi', default = np.random.uniform(-a_b,a_b,self.num_hid)))
        setattr(x,'bf', f.param(name='bf', default = np.random.uniform(-a_b,a_b,self.num_hid)))
        setattr(x,'bo', f.param(name='bo', default = np.random.uniform(-a_b,a_b,self.num_hid)))
        setattr(x,'bc', f.param(name='bc', default = np.random.uniform(-a_b,a_b,self.num_hid)))
        setattr(x,'W2', f.param(name='W2', default = np.random.uniform(-a_w2,a_w2,size=(self.num_hid,self.out_size))))
        setattr(x,'b2', f.param(name='b2', default = np.random.uniform(-a_b,a_b,self.out_size)))


        x.i = list()
        x.cur_c = list()
        x.o = list()
        x.f = list()
        x.h_init = f.input(name='h_init', default=np.zeros((batch_n, self.num_hid)))
        x.c_init = f.input(name='c_init', default=np.zeros((batch_n, self.num_hid)))

        for t in xrange(1, self.max_len+1):
            if t == 1:
                setattr(x, 'i'+str(t), f.sigmoid(f.mul(x.x[t], x.Wi) + f.mul(x.h_init, x.Ui) + x.bi))
                setattr(x, 'f' + str(t), f.sigmoid(
                    f.mul(x.x[t-1], x.Wf) +
                    f.mul(x.h_init, x.Uf) +
                    x.bf))
                setattr(x, 'o' + str(t), f.sigmoid(
                    f.mul(x.x[t-1], x.Wo) +
                    f.mul(x.h_init, x.Uo) +
                    x.bo))
                setattr(x, 'c_tilt' + str(t), f.tanh(
                    f.mul(x.x[t-1], x.Wc) +
                    f.mul(x.h_init, x.Uc) +
                    x.bc))

                setattr(x, 'c'+str(t), f.hadamard(getattr(x, 'f'+str(t)), x.c_init)
                        + f.hadamard(getattr(x, 'i'+str(t)), getattr(x, 'c_tilt'+str(t))) )
                setattr(x, 'h'+str(t), f.hadamard(getattr(x, 'o'+str(t)), f.tanh(getattr(x, 'c'+str(t)))))


            else:    
                setattr(x, 'i'+str(t), f.sigmoid(
                    f.mul(x.x[t-1], x.Wi) +
                    f.mul(getattr(x,'h'+str(t-1)), x.Ui) + x.bi))
                setattr(x, 'f' + str(t), f.sigmoid(
                    f.mul(x.x[t-1], x.Wf) +
                    f.mul(getattr(x, 'h' + str(t - 1)), x.Uf) +
                    x.bf))
                setattr(x, 'o' + str(t), f.sigmoid(
                    f.mul( x.x[t-1], x.Wo) +
                    f.mul(getattr(x, 'h' + str(t - 1)), x.Uo) +
                    x.bo))
                setattr(x, 'c_tilt' + str(t), f.tanh(
                    f.mul(x.x[t-1], x.Wc) +
                    f.mul(getattr(x, 'h' + str(t - 1)), x.Uc) +
                    x.bc))

                setattr(x, 'c'+str(t), f.hadamard(getattr(x, 'f'+str(t)), getattr(x,'c'+str(t-1)))
                        + f.hadamard(getattr(x, 'i'+str(t)), getattr(x, 'c_tilt'+str(t))) )
                setattr(x, 'h'+str(t), f.hadamard(getattr(x, 'o'+str(t)), f.tanh(getattr(x, 'c'+str(t)))))


  

        x.o2 = f.relu(f.mul(getattr(x,'h'+str(self.max_len)), x.W2) + x.b2)
        x.output = f.softMax(x.o2)
        x.loss = f.crossEnt(x.output, x.y)


        return x.setup()

def main(params):
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
    print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    #OPTIONAL: CHECK GRADIENTS HERE


    print "done"

    # train
    print "training..."
    # get default data and params
    value_dict = lstm.my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    w_list = lstm.my_xman.operationSequence(lstm.my_xman.loss)
    ad = Autograd(lstm.my_xman)
    for i in range(epochs):
        for (idxs,e,l) in mb_train:
            #TODO prepare the input and do a fwd-bckwd pass over it and update the weights
            for i in range(1,max_len+1):
                value_dict['x'+str(i)] = np.array(e)[:,max_len-i,:]
            value_dict["y"] = np.array(l)
            value_dict['h_init'] = np.ones((e.shape[0],num_hid))
            value_dict['c_init'] = np.ones((e.shape[0],num_hid))
            value_dict = ad.eval(w_list,value_dict)
            
            updates = ad.bprop(w_list, value_dict, loss=1.0)
            for item in updates:
                if lstm.my_xman.isParam(item):
                    value_dict[item] -= lr * updates[item]
            #save the train loss
            #train_loss = np.append(train_loss, #TODO)
            pass                      
        # validate
        for (idxs,e,l) in mb_valid:
            #TODO prepare the input and do a fwd pass over it to compute the loss
            pass
        #TODO compare current validation loss to minimum validation loss
        # and store params if needed
    print "done"
    #write out the train loss
    np.save(train_loss_file, train_loss)    
    
    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        pass
    #TODO save probabilities on test set
    # ensure that these are in the same order as the test input
    #np.save(output_file, ouput_probabilities)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--train_loss_file', dest='train_loss_file', type=str, default='train_loss')
    params = vars(parser.parse_args())
    main(params)
