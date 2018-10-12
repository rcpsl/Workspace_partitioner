#  * --------------------------------------------------------------------------
#  * File: MultivarMapMotionPlanner.py
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
import constant
from Workspace import Workspace

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

    def __init__(self, nNetwork, num_integrators, workspace, Ts, input_limit):

        self.nNetwork = nNetwork
        
        self.workspace = workspace
        self.obstacles    = workspace.obstacles


        #lasers
        self.laser_angles = workspace.laser_angles
        self.num_lasers   = len(self.laser_angles)

        # Dynamics
        self.num_integrators   = constant.num_integrators
        self.Ts                = Ts
        self.input_constraints = {'ux_max': input_limit, 'ux_min': -1*input_limit, 'uy_max': input_limit, 'uy_min': -1*input_limit}
        
        self.parse()
        
    def parse(self, frm_poly_H_rep, to_poly_H_rep, frm_lidar_config):

        self.frm_poly_H_rep   = frm_poly_H_rep  
        self.to_poly_H_rep    = to_poly_H_rep
        self.frm_lidar_config = frm_lidar_config

        numberOfNeurons = self.nNetwork.nNeurons
        dimOfState = self.num_integrators
        dimOfinput = 2
        image_size = self.nNetwork.inFeaturesLen
        numberOfBoolVars = 0
        numOfRealVars = image_size + (4 * dimOfState) + 2*dimOfinput + (2 * numberOfNeurons)
        numOfConvIFClauses = 2 * numberOfNeurons

        #Create Variables map
        varMap = self.__createVarMap(numOfRealVars, numOfConvIFClauses)

        #instantiate a Solver
        solver = SMConvexSolver.SMConvexSolver(numberOfBoolVars, numOfRealVars, numOfConvIFClauses,
                                    maxNumberOfIterations=10000,
                                    verbose='OFF',  # XS: OFF
                                    profiling='false',
                                    numberOfCores=8,
                                    counterExampleStrategy='IIS',  # XS: IIS
                                    slackTolerance=1E-3)

        # *******************************
        #       Add Constraints
        #********************************
        
        #Add Neural network constraints
        self.__addNNInternalConstraints(solver, varMap)
        print('Added NN Constraints')


        #Add dynamics constraints
        self.add_dynamics_constraints(solver, varMap)
        print('Added Dynamics Constraints')

        #Add Lidar constraints
        self.add_lidar_image_constraints(solver, varMap)        

    #Create a Data structure for mapping solver vars to indices

    def __createVarMap(self, numOfRealVars,numOfIFVars):

        varMap = {}
        varMap['state'] = {}
        varMap['ctrlInput']=[]
        varMap['NN'] = {}
        varMap['bools'] = {}
        inFeaturesLen = self.nNetwork.inFeaturesLen
        dimOfState    = self.num_integrators

        rIdx = 0 
        bIdx = 0

        # Add state indices for state_ and state+
        # Why 12 realVars for the state?

        assert self.num_integrators == 2 #Current varMap data structure only supports 2-integrator
        varMap['current_state'] = {'x': 0, 'y': 1, 'derivatives_x': [2], 'derivatives_y': [3], 'integrator_chain_x': [0, 2], 'integrator_chain_y': [1, 3]}
        varMap['next_state']    = {'x': 4, 'y': 5, 'derivatives_x': [6], 'derivatives_y': [7], 'integrator_chain_x': [4, 6], 'integrator_chain_y': [5, 7]}
        rIdx += 8


        # Add indices for control input
        varMap['ctrlInput']= [rIdx + i for i in range(2)]
        rIdx +=2
        # Add indices for input image
        varMap['image'] = [rIdx + i for i in range(inFeaturesLen)]
        rIdx += inFeaturesLen


        # Add NN nodes indices

        varMap['NN'][0] = {'relu' : varMap['image']}  #Set the first layer of NN to the input image
        for layerKey, layerInfo in self.nNetwork.layers.items():
            varMap['NN'][layerKey] = {}
            lNodes = layerInfo['nNodes']
            varMap['NN'][layerKey]['net']  = [rIdx + i  for i in range(lNodes)]
            rIdx += lNodes
            varMap['NN'][layerKey]['relu']  = [rIdx + i  for i in range(lNodes)]
            rIdx += lNodes
            
            varMap['bools'][layerKey] = np.array([bIdx + i  for i in range(2*lNodes)]).reshape((lNodes,2))

            bIdx += 2*lNodes

        #Add control inputs, they're the the last layer output,so let's take the last 2 indices directly
        varMap['ux'],varMap['uy'] = rIdx-1, rIdx -2
        varMap['ux_norm'], varMap['ux_norm']  = rIdx, rIdx+1 
        rIdx +=2

             

        assert rIdx == numOfRealVars
        assert bIdx == numOfIFVars
        return varMap


        def __addNNInternalConstraints(self, solver, varMap):

        layersKeys = varMap['NN'].keys()
        for layerNum in layersKeys:
            if(layerNum == 0):
                continue
            netVars = varMap['NN'][layerNum]['net']  # Node value before Relu
            prevRelu = varMap['NN'][layerNum-1]['relu']
            weights = self.nNetwork.layers[layerNum]['weights']
            (K, L) = weights.shape
            A = np.block([[np.eye(K),-1 * weights]])
            X = np.concatenate((netVars,prevRelu))
            rVars = [solver.rVars[i] for i in X]
            b = np.zeros(K)

            NetConstraint = SMConvexSolver.LPClause(A, b ,rVars , sense="E")
            solver.addConvConstraint(NetConstraint)

            # Add Boolean constraints
            boolVars = varMap['bools'][layerNum]
            # Prepare the constraint 
            M1 = np.array([
                [1,-1],
                [-1, 1],
                [0, 1] ])

            M2 = np.array([
                [1, 0],
                [-1,0],
                [0, 1] ])

            reluVars  = varMap['NN'][layerNum]['relu']       #Node value after Relu
            for neuron in range(boolVars.shape[0]): #For each node in the layer
                X = [solver.rVars[reluVars[neuron]],solver.rVars[netVars[neuron]] ]

                reluConstraint = SMConvexSolver.LPClause(M1, [0,0,0] ,X, sense ="L")
                solver.setConvIFClause(reluConstraint, boolVars[neuron][0])
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

        def add_lidar_image_constraints(self, solver, varMap):          
        """
        For a certain laser i, if it intersects a vertical obstacle:
            x_i = x_obstacle
            y_i = y_car + (x_obstacle - x_car) tan(laser_angle)
        Otherwise:
            x_i = x_car + (y_obstacle - y_car) cot(laser_angle)
            y_i = y_obstacle
        """
        lidar_config = self.frm_lidar_config
        #print lidar_config

        for i in xrange(self.num_lasers):
            # NOTE: Difference between indices of x,y coordinates for the same laser in image is number of lasers
            rVars = [solver.rVars[varMap['image'][i]], solver.rVars[varMap['image'][i+self.num_lasers]], 
                    solver.rVars[varMap['current_state']['x']], solver.rVars[varMap['current_state']['y']]]

            placement = self.obstacles[lidar_config[i]][4]
            angle     = self.laser_angles[i]

            # TODO: tan, cot do work for horizontal and vertical lasers.
            # TODO: Convert angles to radians out of loop.
            # TODO: Better way to compute cot, maybe numpy.
            if placement: # obstacle is vertical
                obst_x    = self.obstacles[lidar_config[i]][0]
                tan_angle = math.tan(math.radians(angle))
                A = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, tan_angle, -1.0]]
                b = [obst_x, obst_x * tan_angle]

            else: # obstacle is horizontal
                obst_y    = self.obstacles[lidar_config[i]][1]
                cot_angle = math.cos(math.radians(angle)) / math.sin(math.radians(angle))
                A = [[1.0, 0.0, -1.0, cot_angle], [0.0, 1.0, 0.0, 0.0]]     
                b = [obst_y * cot_angle, obst_y]

            image_constraint = SMConvexSolver.LPClause(np.array(A), b, rVars, sense='E') 
            solver.addConvConstraint(image_constraint)

    def add_dynamics_constraints(self, solver, varMap):

        current_integrator_chain_x = varMap['current_state']['integrator_chain_x']
        current_integrator_chain_y = varMap['current_state']['integrator_chain_y']
        next_integrator_chain_x    = varMap['next_state']['integrator_chain_x']
        next_integrator_chain_y    = varMap['next_state']['integrator_chain_y']

        for index in xrange(self.num_integrators - 1):
            # x integrators
            rVars = [solver.rVars[next_integrator_chain_x[index]], 
                     solver.rVars[current_integrator_chain_x[index]], 
                     solver.rVars[current_integrator_chain_x[index+1]]
                    ]
            #print 'chainX', rVars
            dynamics_constraint = SMConvexSolver.LPClause(np.array([[1.0, -1.0, -1*self.Ts]]), [0.0], rVars, sense="E")
            solver.addConvConstraint(dynamics_constraint)

            # y integrators
            rVars = [solver.rVars[next_integrator_chain_y[index]], 
                     solver.rVars[current_integrator_chain_y[index]], 
                     solver.rVars[current_integrator_chain_y[index+1]]
                    ]
            #print 'chainY', rVars
            dynamics_constraints = SMConvexSolver.LPClause(np.array([[1.0, -1.0, -1*self.Ts]]), [0.0], rVars, sense="E")
            solver.addConvConstraint(dynamics_constraints)


        # x last state, such as vx(t+1) = vx(t) + Ts * ux(t)
        rVars = [solver.rVars[next_integrator_chain_x[-1]],
                 solver.rVars[current_integrator_chain_x[-1]],
                 solver.rVars[varMap['ux']]
                ]
        #print 'chainX last', rVars          
        dynamics_constraints = SMConvexSolver.LPClause(np.array([[1.0, -1.0, -1*self.Ts]]), [0.0], rVars, sense="E")
        solver.addConvConstraint(dynamics_constraints) 

        # y last state, such as vy(t+1) = vy(t) + Ts * uy(t)
        rVars = [solver.rVars[next_integrator_chain_y[-1]],
                 solver.rVars[current_integrator_chain_y[-1]],
                 solver.rVars[varMap['uy']]
                ]
        #print 'chainY last', rVars          
        dynamics_constraints = SMConvexSolver.LPClause(np.array([[1.0, -1.0, -1*self.Ts]]), [0.0], rVars, sense="E")
        solver.addConvConstraint(dynamics_constraints) 

    def add_initial_state_constraints(self, solver, varMap):
        # Initial position is in the given subdivision
        A, b = self.frm_poly_H_rep['A'], self.frm_poly_H_rep['b']
        #print 'A = ', A
        #print 'b = ', b
        rVars = [solver.rVars[varMap['current_state']['x']], solver.rVars[varMap['current_state']['y']]]
        position_constraint = SMConvexSolver.LPClause(np.array(A), b, rVars, sense='L')
        #print isinstance(A, list)
        solver.addConvConstraint(position_constraint)

        # TODO: It does not make sense to constraint higher order derivatives to zero in a multi-step scenario
        derivatives = varMap['current_state']['derivatives_x'] + varMap['current_state']['derivatives_y']
        for derivative in derivatives:
            derivative_constraint = SMConvexSolver.LPClause(np.array([[1.0]]), [0.0], [solver.rVars[derivative]], sense='E')
            solver.addConvConstraint(derivative_constraint)

if __name__ == '__main__':

    nn = NeuralNetworkStruct()
    verifier = NN_verifier(nn, 2, Workspace(),constant.Ts,constant.input_limit)
