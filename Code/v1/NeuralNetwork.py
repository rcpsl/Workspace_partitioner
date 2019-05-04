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

    def __init__(self, num_lasers,load_weights = False):

        self.image_size  = 2 * num_lasers
        if(load_weights):
            self.model = load_model("model/my_model.h5")
        # When num_layers is 4: [image_size, hidden_layer_size, hidden_layer_size, hidden_layer_size, last_layer_size]
        layer_sizes = [self.image_size] + [layer.units for layer in self.model.layers]
        self.num_relus = sum(layer_sizes[1:-1])
        self.num_layers = len(self.model.layers)
        print 'Number of Relus: ',self.num_relus
        self.layers = {}
        
        for index in range(len(self.model.layers)):
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
