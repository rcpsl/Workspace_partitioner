#  * --------------------------------------------------------------------------
#  * File: MultiRobotMotionPlanner.py
#  * ---------------------------------------------------------------------------
#  * Copyright (c) 2016 The Regents of the University of California.
#  * All rights reserved.
#  *
#  * Redistribution and use in source and binary forms, with or without
#  * modification, are permitted provided that the following conditions
#  * are met:
#  * 1. Redistributions of source code must retain the above copyright
#  *    notice, this list of conditions and the following disclaimer.
#  * 2. Redistributions in binary form must reproduce the above
#  *    copyright notice, this list of conditions and the following
#  *    disclaimer in the documentation and/or other materials provided
#  *    with the distribution.
#  * 3. All advertising materials mentioning features or use of this
#  *    software must display the following acknowledgement:
#  *       This product includes software developed by Cyber-Physical
#  *       Systems Lab at UCLA and UC Berkeley.
#  * 4. Neither the name of the University nor that of the Laboratory
#  *    may be used to endorse or promote products derived from this
#  *    software without specific prior written permission.
#  *
#  * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS''
#  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#  * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS
#  * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
#  * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#  * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
#  * SUCH DAMAGE.
#  *
#  * Developed by: Yasser Shoukry
#  */

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
from solver import SMConvexSolver

import timeit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from NeuralNetwork import NeuralNetworkStruct

import numpy as np

# ***************************************************************************************************
# ***************************************************************************************************
#
#         CLASS SMConvexSolver
#
# ***************************************************************************************************
# ***************************************************************************************************


class NN_verifier:
    # ========================================================
    #       Constructor
    # ========================================================

    """
    Expects a Neural network structure as follows

    nNetwork{

        'nNeurons'         : int                            #Total number of neurons
        'nLayers'        : int                             #Total number of FC layers
        'inFeaturesLen'    : int                             #Length of the input feature vector
        'layers'         : dictionary                     #Contain NN layers
            {
                '#'     : dictionary                    #indexed with number of the layer > 0, contains all layer info
                {
                    nNodes: int                     #Number of nodes in this layer
                    weights : matrix                 #Weight matrix of the layer
                    .
                    .
                    .

                }

            }

    }

    """

    def __init__(self, nNetwork, numberOfIntegrators, workspace):
        self.numberOfLTLBoolVars = 0
        self.LTL = []
        self.nNetwork = nNetwork
        self.workspace = workspace
        #regions = workspace['regions']
        #self.numberOfRegions = len(regions)
        self.numberOfIntegrators = numberOfIntegrators

        self.__test()
        
    def __test(self):

        numberOfNeurons = self.nNetwork.nNeurons
        dimOfState = self.numberOfIntegrators
        dimOfinput = 2
        inFeaturesLen = self.nNetwork.inFeaturesLen
        numberOfBoolVars = self.numberOfLTLBoolVars  # zero
        numOfRealVars = inFeaturesLen + \
            (2 * dimOfState) + dimOfinput + (2 * numberOfNeurons)
        numOfConvIFClauses = 2 * numberOfNeurons

        optVariables = self.__createOptMap(numOfRealVars, numOfConvIFClauses)

        solver = SMConvexSolver.SMConvexSolver(numberOfBoolVars, numOfRealVars, numOfConvIFClauses,
                                    maxNumberOfIterations=10000,
                                    verbose='OFF',  # XS: OFF
                                    profiling='false',
                                    numberOfCores=8,
                                    counterExampleStrategy='IIS',  # XS: IIS
                                    slackTolerance=1E-3)

        self.__addNNInternalConstraints(solver, optVariables)


    # ***************************************************************************************************
    # ***************************************************************************************************
    #
    #         Add NN constraints
    #
    # ***************************************************************************************************
    # ***************************************************************************************************

    def __addNNInternalConstraints(self, solver, optVariables):

        layersKeys = optVariables['NN'].keys()
        for layerNum in layersKeys:
            if(layerNum == 0):
                continue
            netVars = optVariables['NN'][layerNum]['net']  # Node value before Relu
            prevRelu = optVariables['NN'][layerNum-1]['relu']
            weights = self.nNetwork.layers[layerNum]['weights']
            (K, L) = weights.shape
            A = np.block([[np.eye(K),-1 * weights]])
            X = np.concatenate((netVars,prevRelu))
            rVars = [solver.rVars[i] for i in X]
            b = np.zeros(K)

            NetConstraint = SMConvexSolver.LPClause(A, b ,rVars , sense="E")
            solver.addConvConstraint(NetConstraint)

            # Add Boolean constraints
            boolVars = optVariables['bools'][layerNum]
            # Prepare the constraint 
            M1 = np.array([
                [1,-1],
                [-1, 1],
                [0, 1] ])

            M2 = np.array([
                [1, 0],
                [-1,0],
                [0, 1] ])

            reluVars  = optVariables['NN'][layerNum]['relu']       #Node value after Relu
            for neuron in range(boolVars.shape[0]): #For each node in the layer
                X = [solver.rVars[reluVars[neuron]],solver.rVars[netVars[neuron]] ]

                reluConstraint = SMConvexSolver.LPClause(M1, [0,0,0] ,X, sense ="L")
                print('Created LP clause')
                solver.setConvIFClause(reluConstraint, boolVars[neuron][0])
                print('Added LP clause')

                reluConstraint = SMConvexSolver.LPClause(M2,[0,0,0],[reluVars[neuron],netVars[neuron]], sense ="L")
                solver.setConvIFClause(reluConstraint, boolVars[neuron][1])
                solver.addBoolConstraint(
                    (
                        SMConvexSolver.BoolVar2Int(solver.convIFClauses[ boolVars[neuron][0] ]) 
                        +
                        SMConvexSolver.BoolVar2Int(solver.convIFClauses[ boolVars[neuron][1] ]) 
                    ) 
                        == 1

                    )









    # ***************************************************************************************************
    # ***************************************************************************************************
    #
    #         Create a Data structure for mapping optimization variables
    #
    # ***************************************************************************************************
    # ***************************************************************************************************

    def __createOptMap(self, numOfRealVars,numOfIFVars):

        varMap = {}
        varMap['state'] = {}
        varMap['ctrlInput']=[]
        varMap['NN'] = {}
        varMap['bools'] = {}
        inFeaturesLen = self.nNetwork.inFeaturesLen
        dimOfState    = self.numberOfIntegrators

        rIdx = 0 
        bIdx = 0


        # Add indices for control input
        varMap['ctrlInput']=np.array([rIdx + i for i in range(2)])
        rIdx +=2
        # Add indices for input image
        varMap['input_img'] = np.array([rIdx + i for i in range(inFeaturesLen)])
        rIdx += inFeaturesLen


        # Add state indices for state_ and state+
        varMap['state']['t'] = np.array([rIdx + i for i in range(dimOfState)])
        rIdx += dimOfState
        varMap['state']['t+1'] = np.array([rIdx + i for i in range(dimOfState)])
        rIdx += dimOfState

        # Add NN nodes indices

        varMap['NN'][0] = {'relu' : varMap['input_img']}  #Set the first layer of NN to the input image
        for layerKey, layerInfo in self.nNetwork.layers.items():
            varMap['NN'][layerKey] = {}
            lNodes = layerInfo['nNodes']
            varMap['NN'][layerKey]['net']  = np.array([rIdx + i  for i in range(lNodes)])
            rIdx += lNodes
            varMap['NN'][layerKey]['relu']  = np.array([rIdx + i  for i in range(lNodes)])
            rIdx += lNodes
            
            varMap['bools'][layerKey] = np.array([bIdx + i  for i in range(2*lNodes)]).reshape((lNodes,2))

            bIdx += 2*lNodes


             

                        
        assert rIdx == numOfRealVars
        assert bIdx == numOfIFVars
        return varMap


    



  
if __name__ == '__main__':

    nn = NeuralNetworkStruct()
    verifier = NN_verifier(nn, 2, None)
