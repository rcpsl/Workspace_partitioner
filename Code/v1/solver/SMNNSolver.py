import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/z3/z3-4.4.1-x64-osx-10.11/bin/')
import z3 as z3

import cplex as cplex
from cplex.exceptions import CplexError

import numpy as np
import math

sys.path.insert(0, '../')
from NeuralNetwork import *


class SMNNSolver(object):

    def __init__(self, num_reals, num_relus, max_num_iterations, num_cores, verbose='OFF'):
        
        # ------------ Initialize SAT Solvers ------------------------------------
        self.SATsolver = z3.Solver()
        self.SATsolver.reset()

        # ------------ Initialize Variables --------------------------------------
        self.reals = ['x'+str(i) for i in xrange(num_reals)]
        self.relus = z3.BoolVector('r', num_relus)

        # ------------ Initialize Solver Parameters -------------------------------
        self.max_num_iterations = max_num_iterations
        self.verbose            = verbose
        self.numberOfCores      = num_cores

        # ------------ Initialize Convex Solver  ----------------------------------
        self.convex_solver = cplex.Cplex()
        self.convex_solver.parameters.threads.set(num_cores)

        if self.verbose == 'OFF':
            self.convex_solver.set_results_stream(None)
            self.convex_solver.set_log_stream(None)

        self.convex_solver.objective.set_sense(self.convex_solver.objective.sense.minimize)

        # CPLEX default LB is 0.0, we change it to -infinty. 
        # These variables do not appear in the optimization objective, so we omit the "obj" parameter
        self.convex_solver.variables.add(
                            names   =   self.reals,
                            ub      =   [cplex.infinity] * num_reals,
                            lb      =   [-cplex.infinity] * num_reals
                        )



    def solve(self, robot, num_lasers, trained_nn):     
        # TODO: Only solve optimization with NN constraints when that without NN constraints is feasible

        #solution_found    = False
        iterations_counter = 0

        # ------------ Main Loop ------------------------------------------------------
        while iterations_counter < self.max_num_iterations:
            iterations_counter += 1
            print '********** SMNN Solver, iteration = ', iterations_counter, '****************'

            # ------------ Call SAT solver ------------------------------------------------
            SATcheck = self.SATsolver.check()
            if SATcheck == z3.unsat:
                print '==== Problem is UNSAT ===='
                return list(), list()
                
            else:      
                # ------------ Extract Boolean Models --------------------------------------
                z3_model = self.SATsolver.model()
                relus = [z3.is_true(z3_model[x]) for x in self.relus]
                #print relus

                # ------------ Prepare Convex Problem --------------------------------------
                self.prepare_convex_problem(robot, num_lasers, trained_nn, relus)

                # ------------ Solve Convex Problem ----------------------------------------
                convex_solution_found = self.solve_convex_problem()

                if convex_solution_found == -1 :
                    print '==== ERROR: Cplex Went Wrong ===='
                    return list(), list()

                if convex_solution_found == 1: 
                    print '==== Solution Found ===='
                    reals_model = self.convex_solver.solution.get_values(self.reals)
                    return reals_model, relus_model

                else:
                    # ------------ Find counter examples ----------------------------------------------
                    counter_examples = self.generate_counter_examples(relus)
                    #print 'counter examples = ', counter_examples

                    if not counter_examples: # no counter example can be found .. something is wrong
                        print '==== ERROR: No Counter Example Can Be Found ===='
                        return list(), list()

                    print 'Adding Counter Examples ...'
                    for counter_example in counter_examples:
                        self.SATsolver.add(counter_example)


        print '==== No solution Found: Reached Max Number of Iterations = ', self.max_num_iterations, '====' 
        return list(), list()



    def prepare_convex_problem(self, robot, num_lasers, trained_nn, relus):
        # Compute NN constraints based on current ReLU assignments
        GT, h, QT, c = trained_nn.fc2matrix(relus)
        #print 'GT = ', GT
        #print 'h = ', h
        #print 'QT = ', QT
        #print 'c = ', c
        # TODO: Better way to dynamically manipulate NN constraints instead of adding and deleting for each ReLU assignment.
        # A possible improvement is making a copy of Cplex solver and only add NN constraints to the copy
        self.add_nn_constraints(robot, num_lasers, GT, h, QT, c)



    def add_nn_constraints(self, robot, num_lasers, GT, h, QT, c):
        """
        NN contraints: u = GT d + h^T,  QT d <= c^T
        Each entry of input image d is difference between x or y coordinate of intersection and current car position.
        Re-organize constraint to A b sense c, where b is a list of reals
        """
        # Count number of NN constraints used for deletion
        self.num_nn_constraints = 0
        # Input constraint: u = GT d + h^T
        sums = []
        num_rows = len(h)
        for counter in xrange(num_rows):
            sum1 = -np.sum(GT[counter][:num_lasers])
            sum2 = -np.sum(GT[counter][num_lasers:])
            sums.append([sum1, sum2])
        GT_extended = np.append(GT, sums, axis=1)     
        assert num_rows == 2, 'Only support two control inputs'
        GT_extended = np.append(GT_extended, [[-1.0, 0.0], [0.0, -1.0]], axis=1)
        negative_h  = [-h[i] for i in xrange(num_rows)] 

        reals = self.reals[robot['image'][0]: robot['image'][-1]+1]   
        reals.extend([self.reals[robot['current_state']['x']], self.reals[robot['current_state']['y']], 
                        self.reals[robot['ux']], self.reals[robot['uy']]])
        #print '\nGT_extended = ', GT_extended
        #print 'negative_h = ', negative_h
        #print 'reals = ', reals

        nn_input_constraints = LPClause(GT_extended, negative_h, reals, sense='E')
        self.num_nn_constraints += len(nn_input_constraints['senses'])
        self.add_conv_constraint(nn_input_constraints, 'NNConstraint')


        # ReLU constraint: QT d <= c^T
        sums = []
        num_rows = len(c)
        for counter in xrange(num_rows):
            sum1 = -np.sum(QT[counter][:num_lasers])
            sum2 = -np.sum(QT[counter][num_lasers:])
            sums.append([sum1, sum2])
        QT_extended = np.append(QT, sums, axis=1)
        
        reals = self.reals[robot['image'][0]: robot['image'][-1]+1]   
        reals.extend([self.reals[robot['current_state']['x']], self.reals[robot['current_state']['y']]])
        #print '\nQT_extended = ', QT_extended
        #print 'c = ', c
        #print 'reals = ', reals

        nn_relu_constraints = LPClause(QT_extended, c, reals, sense='L')
        self.num_nn_constraints += len(nn_relu_constraints['senses'])
        self.add_conv_constraint(nn_relu_constraints, 'NNConstraint')



    def solve_convex_problem(self):
        self.convex_solver.solve()
        if self.convex_solver.solution.get_status() == 1:
            convex_solution_found = 1
        elif self.convex_solver.solution.get_status() != 1:
            convex_solution_found = 0
        else:
            convex_solution_found = -1 # something went wrong

        # Delete NN constraints
        # TODO: It should be more efficient to delete constraints by sequence of indices
        for i in xrange(self.num_nn_constraints):
            self.convex_solver.linear_constraints.delete("NNConstraint")
            
        return convex_solution_found



    def generate_counter_examples(self, relus):
        # Trivial counter example strategy
        counter_examples = []
        #active_relu_indices = [i for i, x in enumerate(relus) if x == True]
        #counter_example = z3.Or([self.relus[counter] != relus[counter] for counter in active_relu_indices])
        counter_example = z3.Or([self.relus[counter] != relus[counter] for counter in xrange(len(relus))])
        counter_examples.append(counter_example)
        return counter_examples

   

    # ========================================================
    #
    # Following routines are DIRECTLY COPIED FROM SMC with standardized routine and class attribute names
    #
    # ========================================================
    def add_conv_constraint(self, constraint, name=None):
        # XS: add argument name
        #print constraint['lin_expr']

        if name:
            if constraint['type'] == 'LP':
                names = [name] * len(constraint['senses'])
                self.convex_solver.linear_constraints.add(
                            lin_expr    = constraint['lin_expr'],
                            senses      = constraint['senses'],
                            rhs         = constraint['rhs'],
                            names       = names
                    )
            elif constraint['type'] == 'QP':
                self.convex_solver.quadratic_constraints.add(
                                quad_expr   = constraint['quad_expr'],
                                lin_expr    = constraint['lin_expr'],
                                sense       = constraint['sense'],
                                rhs         = constraint['rhs'],
                                names       = names
                    )
        else:
            if constraint['type'] == 'LP':
                self.convex_solver.linear_constraints.add(
                            lin_expr    = constraint['lin_expr'],
                            senses      = constraint['senses'],
                            rhs         = constraint['rhs'],
                    )
            elif constraint['type'] == 'QP':
                self.convex_solver.quadratic_constraints.add(
                                quad_expr   = constraint['quad_expr'],
                                lin_expr    = constraint['lin_expr'],
                                sense       = constraint['sense'],
                                rhs         = constraint['rhs'],
                    )




# ***************************************************************************************************
# ***************************************************************************************************
#
#                                   Public Helper APIs
#                               DIRECTLY COPIED FROM SMC
#
# ***************************************************************************************************
# ***************************************************************************************************
def LPClause(A, b, rVars, sense = "L"):
    # x = rVars
    # sense = "L", "G", "E"
    #A x {sense} b, A is a matrix and b is a vector with same dimension
    
    # TODO: add a dimension check
    
    # Put the constraint in CPLEX format, example below:
    # lin_expr = [cplex.SparsePair(ind = ["x1", "x3"], val = [1.0, -1.0]),\
    #    cplex.SparsePair(ind = ["x1", "x2"], val = [1.0, 1.0]),\
    #    cplex.SparsePair(ind = ["x1", "x2", "x3"], val = [-1.0] * 3),\
    #    cplex.SparsePair(ind = ["x2", "x3"], val = [10.0, -2.0])],\
    # senses = ["E", "L", "G", "R"],\
    # rhs = [0.0, 1.0, -1.0, 2.0],\

    #XS: A is numpy array, b is list
    #print isinstance(b, list)

    numOfRows       = len(b)
    lin_expr        = list()
    
    for counter in range(0, numOfRows):
        lin_expr.append(cplex.SparsePair(ind = rVars, val = A[counter,:]))
    
    rhs             = b
    senses          = [sense] * numOfRows

    constraint  = {'type':'LP', 'lin_expr':lin_expr, 'rhs':rhs, 'x':rVars, 'senses':senses, 'A':A}
    return constraint