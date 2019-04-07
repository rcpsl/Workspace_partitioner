import numpy as np
import constant
import pickle
import sys
from Workspace import *
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="2";
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)
K.set_session(session)
from keras.models import load_model

class NeuralNetworkStruct(object):

    def __init__(self, num_lasers,num_layers,hidden_layer_size, load_weights = False):
        # num_lasers includes the output layer
        self.num_layers = num_layers # Haitham: this num_layers is one more than that in our reslult form
        self.hidden_layer_size = hidden_layer_size
        last_layer_size = 2

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
        if(load_weights):
            self.model = load_model("model/my_model.h5")
        for index in range(self.num_layers):
            self.layers[index+1]  = {'num_nodes': layer_sizes[index+1], 'weights': []}
            self.layers[index+1]['type'] = 'hidden'

            if load_weights:
                self.layers[index+1]['weights'] = self.model.get_weights()[2*index].T
                self.layers[index+1]['bias'] = self.model.get_weights()[2*index + 1]


            else:
                self.layers[index+1]['weights'] = np.random.normal(scale=2.0, size=(layer_sizes[index+1], layer_sizes[index]))
                self.layers[index+1]['bias'] = np.random.normal(scale=2.0, size=(layer_sizes[index+1]))

            #print self.layers[index+1]['weights']

        self.layers[self.num_layers]['type'] = 'output'


"""
if __name__ == '__main__':
    trained_nn = NeuralNetworkStruct(4)
    print trained_nn.layer_sizes
"""    
