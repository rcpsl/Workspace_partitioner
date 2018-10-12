import numpy as np
import constant
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
    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg


    def __init__(self):
        
        self.inFeaturesLen = 2 * (constant.num_of_laser)
        self.nLayers = 4
        self.nNeurons = 150*3 + 2
        self.nRelus = 150*3
        layer_size = 150
        self.layers_size = [self.inFeaturesLen, layer_size, layer_size, layer_size,2]

        self.layers = {}

        for i in range(self.nLayers):
            self.layers[i+1]  = {'nNodes' : self.layers_size[i+1],'weights':[]}
            self.layers[i+1]['weights'] = np.random.randn(self.layers_size[i+1],self.layers_size[i])
            self.layers[i+1]['type'] = 'hidden'
        
        self.layers[self.nLayers]['type'] = 'output'


if __name__ == '__main__':

    NN = NeuralNetworkStruct()