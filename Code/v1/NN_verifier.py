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
import pickle

import numpy as np
import math
import argparse
import sys

# ***************************************************************************************************
# ***************************************************************************************************
#
#         CLASS SMConvexSolver
#
# ***************************************************************************************************
# ***************************************************************************************************
class NN_verifier:

    def __init__(self, trained_nn, workspace, num_integrators, Ts, higher_deriv_bound,out_file = ''):
        # Workspace
        self.laser_angles = workspace.laser_angles
        self.num_lasers   = len(self.laser_angles)
        self.obstacles    = workspace.obstacles

        # Dynamics
        self.num_integrators    = num_integrators
        self.num_control_inputs = 2
        self.Ts                 = Ts
        self.higher_deriv_bound = higher_deriv_bound
        #self.input_constraints = {'ux_max': input_limit, 'ux_min': -1*input_limit, 'uy_max': input_limit, 'uy_min': -1*input_limit}
        
        # Trained NN
        self.trained_nn = trained_nn
        self.num_relus  = trained_nn.num_relus
        self.image_size = trained_nn.image_size
        assert self.image_size == 2 * self.num_lasers, 'Image size should be twice of number of lasers'
        self.out_file = out_file


    def parse(self, frm_poly_H_rep, to_poly_H_rep, frm_lidar_config, frm_abst_index, frm_refined_index,
              preprocess=True, use_ctr_examples = True, max_iter = 10000,verbose = 'OFF'):
        self.frm_poly_H_rep   = frm_poly_H_rep
        self.to_poly_H_rep    = to_poly_H_rep
        self.frm_lidar_config = frm_lidar_config


        #numOfRealVars = self.image_size + (4 * self.num_integrators) + 2 * self.num_control_inputs + \
        #                (2 * self.num_relus + self.num_control_inputs)
        numOfRealVars = self.image_size + (4 * self.num_integrators) + (2 * self.num_relus + self.num_control_inputs)   
        numOfConvIFClauses = 2 * self.num_relus

        #Create Variables map
        varMap = self.__createVarMap(numOfRealVars, numOfConvIFClauses)
        #print varMap

        #instantiate a Solver
        numberOfBoolVars = 0
        solver = SMConvexSolver.SMConvexSolver(numberOfBoolVars, numOfRealVars, numOfConvIFClauses,
                                    maxNumberOfIterations= max_iter,
                                    verbose= verbose,  # XS: OFF
                                    profiling='false',
                                    numberOfCores=8,
                                    counterExampleStrategy='IIS',  # XS: IIS
                                    slackTolerance=1E-3)

        # Mode and parameters
        # preprocess       = False
        # use_ctr_examples = True

        fname = 'counterexamples/layers_' + str(self.trained_nn.num_layers) + \
                '_neurons_' + str(self.trained_nn.hidden_layer_size) + \
                '_region_' + str(frm_abst_index) + '-' + str(frm_refined_index)


        #Add Neural network constraints
        self.__addNNInternalConstraints(solver, varMap)

        #Add Lidar constraints
        self.add_lidar_image_constraints(solver, varMap) 

        #Add initial state constraints
        self.add_initial_state_constraints(solver, varMap)

        if preprocess == False:
            #Add dynamics constraints
            self.add_dynamics_constraints(solver, varMap)

            #Add goal state constaints
            self.add_goal_state_constraints(solver, varMap)


        # Check transition: Load CE
        if (preprocess == False) and (use_ctr_examples == True):
            with open(fname,'rb') as f:
                counter_examples = pickle.load(f)
                f.close()
            #print 'Load CEs = ', counter_examples     
            print 'Number of loaded CEs = ', len(counter_examples)  
            self.__add_counter_examples(solver, counter_examples)


        # SMC solve
        # print '\nFirst SMC solve'
        start_time = timeit.default_timer()
        rVarsModel, bModel, convIFModel = solver.solve()
        end_time   = timeit.default_timer()
        #print 'New CEs = ', solver.counterExamples
        # print 'Number of new CEs = ', len(solver.counterExamples)


        # Preprocess: Generate and store CEs
        if preprocess == True:

            # Add solution to counter examples
            indices_of_solution_ce = []
            count_iters = 0 
            while convIFModel:
                count_iters += 1
                # print '\nAdditional SMC solve, ', count_iters
                # Add solution as an counter example. The index is recorded in order to pick out later
                indices_of_solution_ce.append(len(solver.counterExamples))
                constraint = [solver.convIFClauses[counter] != convIFModel[counter] for counter in xrange(numOfConvIFClauses)]
                solver.addBoolConstraint(SMConvexSolver.OR(*constraint))

                rVarsModel, bModel, convIFModel = solver.solve()
                # NOTE: Why this number sometimes is one less than the loaded number of CEs?
                if(verbose == 'ON'):
                    print 'Accumulative number of CEs = ', len(solver.counterExamples) - len(indices_of_solution_ce)
                    print '\n'
            end_time   = timeit.default_timer()
            cumulative_CE = len(solver.counterExamples) - len(indices_of_solution_ce)

            # Pick out CEs correspond to solution
            # NOTE: Even with solution stored as a CE, solution may still be found when load the CEs.
            # This is because solution could be either True or False, while CEs are NOT True while loading.
            solver.counterExamples = [x for i, x in enumerate(solver.counterExamples) if i not in indices_of_solution_ce]
            with open(fname, 'wb') as f:
                pickle.dump(solver.counterExamples, f)
                f.close()
            with open(fname + '.txt', 'wb') as f:
                for example in solver.counterExamples:
                    for bool_var_index in example:
                        f.write("%s " % bool_var_index)
                    f.write("\n")
                f.close()
        
        if(preprocess):
            # print('#Hidden Layers\t#Neurons\t#SAT Assignments\t#CE\tTime(s)')
            if(len(self.out_file) > 0):
                if('table2' in self.out_file):
                    f = open(self.out_file,'a+')
                    f.write('\t' + str(self.trained_nn.num_layers-1) + '\t  ' + str(self.trained_nn.num_neurons) + '\t\t\t' + str(count_iters) +'\t\t'+ str(cumulative_CE) +'\t%.5f'%(end_time - start_time) + '\n')
                    f.write('------------------------------------------------------------------------\n')
                    f.close()
                else:
                    f = open(self.out_file,'a+')
                    f.write('\t\t\t' + str(count_iters) +'\t\t'+ str(cumulative_CE) +'\t%.5f'%(end_time - start_time) + '\n')
                    f.write('------------------------------------------------------------------------\n')
                    f.close()
            print '\t' + str(self.trained_nn.num_layers-1) + '\t  ' + str(self.trained_nn.num_neurons) + '\t\t\t' + str(count_iters) +'\t\t'+ str(cumulative_CE) +'\t%.5f'%(end_time - start_time)
        else:
            if(len(rVarsModel) == 0):
                print("============NO Solution===============")
                print("%.5f"%(end_time - start_time))

            else:
                print("=======Solution found========")
                print 'Layers' + '\t' + 'hidden' + '\t' + 'neurons' + '\tTime'
                print str(self.trained_nn.num_layers) + '\t' + str(self.trained_nn.hidden_layer_size) + '\t' + str(self.trained_nn.num_neurons) +'\t%.5f'%(end_time - start_time)
            if(len(self.out_file) > 0):
                f = open(self.out_file,'a+')
                f.write('\t\t\t%3.5f'%(end_time - start_time))
                f.close()

    def __createVarMap(self, numOfRealVars,numOfIFVars):
        """
        Create a Data structure for mapping solver vars to indices
        """
        varMap = {}
        #varMap['state'] = {}
        varMap['ctrlInput']=[]
        varMap['NN'] = {}
        varMap['bools'] = {}
        inFeaturesLen = self.image_size
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
        #varMap['ctrlInput']= [rIdx + i for i in range(2)]
        #rIdx +=2
        # Add indices for input image
        varMap['image'] = [rIdx + i for i in range(inFeaturesLen)]
        rIdx += inFeaturesLen


        # Add NN nodes indices

        varMap['NN'][0] = {'relu': varMap['image']}  #Set the first layer of NN to the input image
        for layerKey, layerInfo in self.trained_nn.layers.items():
            varMap['NN'][layerKey] = {}
            lNodes = layerInfo['num_nodes']
            varMap['NN'][layerKey]['net']  = [rIdx + i  for i in range(lNodes)]
            rIdx += lNodes
            if(self.trained_nn.layers[layerKey]['type'] == 'hidden'):
                varMap['NN'][layerKey]['relu']  = [rIdx + i  for i in range(lNodes)]
                rIdx += lNodes
            else:
                #This part is just for consistency of keys in the dictionary (setting lastLayer['Relu'] = lastLayer['net'])
                varMap['NN'][layerKey]['relu'] = varMap['NN'][layerKey]['net']

            
            if(self.trained_nn.layers[layerKey]['type'] == 'hidden'): #No need for Boolean variables for outputs
                varMap['bools'][layerKey] = np.array([bIdx + i  for i in range(2*lNodes)]).reshape((lNodes,2))
                bIdx += 2*lNodes

        #Add control inputs, they're the last layer output,so let's take the last 2 indices directly
        # XS: order is wrong
        #varMap['ux'],varMap['uy'] = rIdx-1, rIdx -2
        varMap['ux'], varMap['uy'] = rIdx-2, rIdx -1
        # XS: Why both ux_norm
        #varMap['ux_norm'], varMap['ux_norm'] = rIdx, rIdx+1 
        #rIdx +=2

             
        # print(rIdx,numOfRealVars)
        # print(bIdx, numOfIFVars)
        assert rIdx == numOfRealVars
        assert bIdx == numOfIFVars
        return varMap



    def __addNNInternalConstraints(self, solver, varMap):
        nLayers = varMap['NN'].keys()
        lastLayer = nLayers[-1]

        for layerNum in nLayers:
            if(layerNum == 0):   #Add these constraints only for the Hidden layers
                continue
            netVars = varMap['NN'][layerNum]['net']  # Node value before Relu
            prevRelu = varMap['NN'][layerNum-1]['relu']

            weights = self.trained_nn.layers[layerNum]['weights']
            (K, L) = weights.shape
            A = np.block([[np.eye(K), -1 * weights]])
            X = np.concatenate((netVars, prevRelu))
            rVars = [solver.rVars[i] for i in X]
            b = np.zeros(K)

            NetConstraint = SMConvexSolver.LPClause(A, b, rVars, sense="E")
            solver.addConvConstraint(NetConstraint)


            # Prepare the boolean constraint only for hidden layers
            if(self.trained_nn.layers[layerNum]['type'] == 'hidden'):        
                boolVars = varMap['bools'][layerNum] #2D array[node][0/1] before and after Relu
                M1 = np.array([
                    [1.0, -1.0],
                    [-1.0, 1.0],
                    [0.0, -1.0] ])

                M2 = np.array([
                    [1.0, 0.0],
                    [-1.0, 0.0],
                    [0.0, 1.0] ])

                reluVars  = varMap['NN'][layerNum]['relu']       #Node value after Relu
                for neuron in range(boolVars.shape[0]): #For each node in the layer
                    X = [solver.rVars[reluVars[neuron]],solver.rVars[netVars[neuron]] ]

                    #reluConstraint = SMConvexSolver.LPClause(M1, [0,0,0] ,X, sense ="L")
                    reluConstraint = SMConvexSolver.LPClause(M1, [0.0, 0.0, 0.0] ,X, sense ="L")
                    solver.setConvIFClause(reluConstraint, boolVars[neuron][0])
                    #reluConstraint = SMConvexSolver.LPClause(M2,[0,0,0],X, sense ="L")
                    reluConstraint = SMConvexSolver.LPClause(M2,[0.0, 0.0, 0.0],X, sense ="L")
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
                A = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, tan_angle, 0.0]]
                #A = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, tan_angle, -1.0]]
                b = [obst_x, obst_x * tan_angle]

            else: # obstacle is horizontal
                obst_y    = self.obstacles[lidar_config[i]][1]
                cot_angle = math.cos(math.radians(angle)) / math.sin(math.radians(angle))
                A = [[1.0, 0.0, 0.0, cot_angle], [0.0, 1.0, 0.0, 1.0]]     
                #A = [[1.0, 0.0, -1.0, cot_angle], [0.0, 1.0, 0.0, 0.0]]     
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
            # derivative_constraint = SMConvexSolver.LPClause(np.array([[1.0]]), [0.0], [solver.rVars[derivative]], sense='E')
            derivative_constraint = SMConvexSolver.LPClause(np.array([[1.0]]), [self.higher_deriv_bound], [solver.rVars[derivative]], sense='L')
            derivative_constraint = SMConvexSolver.LPClause(np.array([[1.0]]), [-1 * self.higher_deriv_bound], [solver.rVars[derivative]], sense='G')
            solver.addConvConstraint(derivative_constraint)



    def add_goal_state_constraints(self, solver, varMap):
        # Goal position is in the given subdivision
        A, b = self.to_poly_H_rep['A'], self.to_poly_H_rep['b']
        rVars = [solver.rVars[varMap['next_state']['x']], solver.rVars[varMap['next_state']['y']]]
        position_constraint = SMConvexSolver.LPClause(np.array(A), b, rVars, sense='L')
        solver.addConvConstraint(position_constraint)
    
        # TODO: It does not make sense to constraint higher order derivatives to zero in a multi-step scenario
        derivatives = varMap['next_state']['derivatives_x'] + varMap['next_state']['derivatives_y']
        for derivative in derivatives:
            # derivative_constraint = SMConvexSolver.LPClause(np.array([[1.0]]), [0.0], [solver.rVars[derivative]], sense='E')
            derivative_constraint = SMConvexSolver.LPClause(np.array([[1.0]]), [self.higher_deriv_bound], [solver.rVars[derivative]], sense='L')
            derivative_constraint = SMConvexSolver.LPClause(np.array([[1.0]]), [-1 * self.higher_deriv_bound], [solver.rVars[derivative]], sense='G')   
            solver.addConvConstraint(derivative_constraint)
    


    def __add_counter_examples(self ,solver, counter_examples):
        for example in counter_examples:
            constraint = [SMConvexSolver.NOT(solver.convIFClauses[idx]) for idx in example]
            solver.addBoolConstraint(SMConvexSolver.OR(*constraint))
