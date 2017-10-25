#from models import rbm
from dataio import txt_to_array
import numpy as np
import pdb
from models import RBM
import matplotlib.pyplot as plt


def main():
    train_data = txt_to_array('digitstrain.txt')
    val_data = txt_to_array('digitsvalid.txt')
    n_visible = train_data.shape[1]
    
    
    for k in [10,20]:
        rbm = RBM(n_visible,100)
        rbm.train(train_data,val_data,k,50)
        plt.plot(rbm.epoch_nums,rbm.avg_train_loss, 'r--', rbm.avg_val_loss, 'b--')
        plt.savefig('q5b{0}.png'.format(k))
        plt.close()
        np.save('weights{0}'.format(k),rbm.W)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='script to convert raw train') 
    
    parser.add_argument('-e', '--epath',  type=str, default= "test.txt", help='path to input era path')
    parser.add_argument('-p', '--ppath',  type=str, default= "test.txt", help='path to input prediction path')
    parser.add_argument('-l', '--lpath',  type=str, default= "test.txt", help='path to input labels path')
    parser.add_argument('-i', '--ipath',  type=str, default= "test.txt", help='path to output consistency path')
    
    args = parser.parse_args()

    main()