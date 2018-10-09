import numpy as np
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
        
        self.inFeaturesLen = 10
        self.nLayers = 3
        self.nNeurons = 30
        layer_size = 10
        self.layers_size = [self.inFeaturesLen, layer_size, layer_size, layer_size]

        self.layers = {}

        for i in range(self.nLayers):
            self.layers[i]  = {'nNodes' : self.layers_size[i+1],'weights':[]}
            self.layers[i]['weights'] = np.random.randn(self.layers_size[i+1],self.layers_size[i])
        print(self.layers)


if __name__ == '__main__':

    NN = NeuralNetworkStruct()