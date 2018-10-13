import numpy as np
import constant
import pickle
import sys

"""
Expects a Neural network structure as follows

nNetwork{

    'nNeurons'      : int                           #Total number of neurons
    'nLayers'       : int                           #Total number of FC layers
    'inFeaturesLen' : int                           #Length of the input feature vector
    'layers'        : dictionary                    #Contain NN layers
        {
            '#'     : dictionary                    #indexed with number of the layer > 0, contains all layer info
            {
                nNodes: int                     #Number of nodes in this layer
                weights : matrix                #Weight matrix of the layer
                .
                .
                .

            }
        
        } 

}

"""
class NeuralNetworkStruct(object):

    def __init__(self, layer_size,load_weights = False):
        
        self.inFeaturesLen = 2 * (constant.num_of_laser)
        self.nLayers = 4
        self.layer_size = layer_size
        self.nNeurons = self.layer_size * (self.nLayers-1) + 2
        self.nRelus = self.layer_size * (self.nLayers-1)
        self.layers_size = [self.inFeaturesLen, self.layer_size, self.layer_size, self.layer_size,2]
        self.__weight_files = ['weights/0_w_in_FC1','weights/1_w_FC1_FC2','weights/2_w_FC2_FC3','weights/3_w_FC3_out']

       
        self.layers = {}
        for i in range(self.nLayers):
            self.layers[i+1]  = {'nNodes' : self.layers_size[i+1],'weights':[]}
            self.layers[i+1]['type'] = 'hidden'
            if(load_weights and self.layer_size == 200):
                with open(self.__weight_files[i]) as f:
                    weights = pickle.load(f)
                self.layers[i+1]['weights'] = weights
            else:
                self.layers[i+1]['weights'] = np.random.normal(scale = 2.0,size = (self.layers_size[i+1],self.layers_size[i]))

        
        self.layers[self.nLayers]['type'] = 'output'


if __name__ == '__main__':

    NN = NeuralNetworkStruct()