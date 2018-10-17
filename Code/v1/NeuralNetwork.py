import numpy as np
import constant
import pickle
import sys
from Workspace import *

class NeuralNetworkStruct(object):

    def __init__(self, num_lasers,num_layers,hidden_layer_size):
        # num_lasers includes the output layer
        self.num_layers = num_layers # Haitham: this num_layers is one more than that in our reslult form
        self.hidden_layer_size = hidden_layer_size
        last_layer_size = 2
        load_weights = False 

        self.num_relus   = self.hidden_layer_size * (self.num_layers-1)
        self.num_neurons = self.num_relus + last_layer_size
        self.image_size  = 2 * num_lasers

        print 'Number of neurons = ', self.num_neurons

        # When num_layers is 4: [image_size, hidden_layer_size, hidden_layer_size, hidden_layer_size, last_layer_size]
        layer_sizes = [self.image_size]
        for index in xrange(self.num_layers-1):
            layer_sizes.append(self.hidden_layer_size)
        layer_sizes.append(last_layer_size)     

        weight_files = ['weights/w_in_FC1','weights/w_FC1_FC2','weights/w_FC2_FC3','weights/w_FC3_out']

        self.layers = {}
        for index in range(self.num_layers):
            self.layers[index+1]  = {'num_nodes': layer_sizes[index+1], 'weights': []}
            self.layers[index+1]['type'] = 'hidden'

            if load_weights:
                with open(weight_files[index]) as f:
                    weights = pickle.load(f)
                self.layers[index+1]['weights'] = weights
            else:
                self.layers[index+1]['weights'] = np.random.normal(scale=2.0, size=(layer_sizes[index+1], layer_sizes[index]))

            #print self.layers[index+1]['weights']

        self.layers[self.num_layers]['type'] = 'output'


"""
if __name__ == '__main__':
    trained_nn = NeuralNetworkStruct(4)
    print trained_nn.layer_sizes
"""    