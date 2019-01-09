import sys
sys.path.insert(0, './solver')
from SMNNSolver import *


class VerifyNNParser(object):

    def __init__(self, workspace, trained_nn, num_integrators, Ts, input_limit):
        # Workspace
        self.laser_angles = workspace.laser_angles
        self.num_lasers   = len(self.laser_angles)
        self.obstacles    = workspace.obstacles

        # Trained NN
        self.trained_nn = trained_nn
        self.num_relus  = trained_nn.num_relus 
        self.image_size = trained_nn.image_size 
        assert self.image_size == 2 * self.num_lasers, 'Image size should be twice of number of lasers'

        # Dynamics
        self.num_integrators   = num_integrators
        self.Ts                = Ts
        self.input_constraints = {'ux_max': input_limit, 'ux_min': -1*input_limit, 'uy_max': input_limit, 'uy_min': -1*input_limit}
                                      


    def parse(self, frm_poly_H_rep, to_poly_H_rep, frm_lidar_config):
        self.frm_poly_H_rep   = frm_poly_H_rep
        self.to_poly_H_rep    = to_poly_H_rep
        self.frm_lidar_config = frm_lidar_config

        # Create robot data structure
        robot, num_reals = self.create_robot_data_structure()
        print 'Number of reals = ', num_reals
        print 'Number of ReLUs = ', self.num_relus

        # Initialize SMNN solver
        solver = SMNNSolver(num_reals, self.num_relus, max_num_iterations=1, num_cores=4, verbose='OFF')

        # Add initial state constraints
        self.add_initial_state_constraints(solver, robot)

        # Add goal constraints
        self.add_goal_state_constraints(solver, robot)

        # Add LiDAR image constraints
        self.add_lidar_image_constraints(solver, robot)

        # Add dynamics constraints
        self.add_dynamics_constraints(solver, robot)

        # Add input constraints
        self.add_input_constraints(solver, robot)

        # Currently, NN constraints are added and deleted dynamically based on SAT solver results in SMNNSolver

        # SMNN solve
        reals_model, relus_model = solver.solve(robot, self.num_lasers, self.trained_nn)

        #print 'Number of linear constraints = ', solver.convex_solver.linear_constraints.get_num()
        #print 'Name of linear constraints: ', solver.convex_solver.linear_constraints.get_names()
        #print solver.convex_solver.linear_constraints.get_rows()
        #print solver.convex_solver.linear_constraints.get_rhs()


        if not reals_model:
            return list()

        # TODO: Collect real variables if solution is found



    def create_robot_data_structure(self):
        # TODO: Beyond 2-integrator
        assert self.num_integrators == 2, 'Current robot data structure only supports 2-integrator'
        current_state = {'x': 0, 'y': 1, 'derivatives_x': [2], 'derivatives_y': [3], 'integrator_chain_x': [0, 2], 'integrator_chain_y': [1, 3]}
        next_state    = {'x': 4, 'y': 5, 'derivatives_x': [6], 'derivatives_y': [7], 'integrator_chain_x': [4, 6], 'integrator_chain_y': [5, 7]}
        num_state_indices = 12

        # NOTE: Assume LiDAR image is in the form [x0-xc, x1-xc,..., y0-yc, y1-yc,...]^T,
        # then 'image' below stores indices for variables in order x0, x1,..., y0, y1,...
        image = range(num_state_indices, num_state_indices+self.image_size)

        robot = {'current_state': current_state, 'next_state': next_state, 'ux': 8, 'uy': 9, 
                'ux_norm': 10, 'uy_norm': 11, 'image': image}
        num_reals  = num_state_indices + self.image_size

        return robot, num_reals



    def add_initial_state_constraints(self, solver, robot):
        # Initial position is in the given subdivision
        A, b = self.frm_poly_H_rep['A'], self.frm_poly_H_rep['b']
        #print 'A = ', A
        #print 'b = ', b
        reals = [solver.reals[robot['current_state']['x']], solver.reals[robot['current_state']['y']]]
        position_constraint = LPClause(np.array(A), b, reals, sense='L')
        #print isinstance(A, list)
        solver.add_conv_constraint(position_constraint)

        # TODO: It does not make sense to constraint higher order derivatives to zero in a multi-step scenario
        derivatives = robot['current_state']['derivatives_x'] + robot['current_state']['derivatives_y']
        for derivative in derivatives:
            derivative_constraint = LPClause(np.array([[1.0]]), [0.0], [solver.reals[derivative]], sense='E')
            solver.add_conv_constraint(derivative_constraint)



    def add_goal_state_constraints(self, solver, robot):
        # Goal position is in the given subdivision
        A, b = self.to_poly_H_rep['A'], self.to_poly_H_rep['b']
        reals = [solver.reals[robot['next_state']['x']], solver.reals[robot['next_state']['y']]]
        position_constraint = LPClause(np.array(A), b, reals, sense='L')
        solver.add_conv_constraint(position_constraint)
    
        # TODO: It does not make sense to constraint higher order derivatives to zero in a multi-step scenario
        derivatives = robot['next_state']['derivatives_x'] + robot['next_state']['derivatives_y']
        for derivative in derivatives:
            derivative_constraint = LPClause(np.array([[1.0]]), [0.0], [solver.reals[derivative]], sense='E')
            solver.add_conv_constraint(derivative_constraint)



    def add_lidar_image_constraints(self, solver, robot):          
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
            reals = [solver.reals[robot['image'][i]], solver.reals[robot['image'][i+self.num_lasers]], 
                    solver.reals[robot['current_state']['x']], solver.reals[robot['current_state']['y']]]

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

            image_constraint = LPClause(np.array(A), b, reals, sense='E') 
            solver.add_conv_constraint(image_constraint)

        

    def add_dynamics_constraints(self, solver, robot):
        # state_i(t+1) = state_i(t) + Ts * state_{i + 1}(t)
        # TODO: Test correctness for high order integrator
        current_integrator_chain_x = robot['current_state']['integrator_chain_x']
        current_integrator_chain_y = robot['current_state']['integrator_chain_y']
        next_integrator_chain_x    = robot['next_state']['integrator_chain_x']
        next_integrator_chain_y    = robot['next_state']['integrator_chain_y']

        for index in xrange(self.num_integrators - 1):
            # x integrators
            reals = [solver.reals[next_integrator_chain_x[index]], 
                     solver.reals[current_integrator_chain_x[index]], 
                     solver.reals[current_integrator_chain_x[index+1]]
                    ]
            #print 'chainX', reals
            dynamics_constraint = LPClause(np.array([[1.0, -1.0, -1*self.Ts]]), [0.0], reals, sense="E")
            solver.add_conv_constraint(dynamics_constraint)

            # y integrators
            reals = [solver.reals[next_integrator_chain_y[index]], 
                     solver.reals[current_integrator_chain_y[index]], 
                     solver.reals[current_integrator_chain_y[index+1]]
                    ]
            #print 'chainY', reals
            dynamics_constraints = LPClause(np.array([[1.0, -1.0, -1*self.Ts]]), [0.0], reals, sense="E")
            solver.add_conv_constraint(dynamics_constraints)


        # x last state, such as vx(t+1) = vx(t) + Ts * ux(t)
        reals = [solver.reals[next_integrator_chain_x[-1]],
                 solver.reals[current_integrator_chain_x[-1]],
                 solver.reals[robot['ux']]
                ]
        #print 'chainX last', reals          
        dynamics_constraints = LPClause(np.array([[1.0, -1.0, -1*self.Ts]]), [0.0], reals, sense="E")
        solver.add_conv_constraint(dynamics_constraints) 

        # y last state, such as vy(t+1) = vy(t) + Ts * uy(t)
        reals = [solver.reals[next_integrator_chain_y[-1]],
                 solver.reals[current_integrator_chain_y[-1]],
                 solver.reals[robot['uy']]
                ]
        #print 'chainY last', reals          
        dynamics_constraints = LPClause(np.array([[1.0, -1.0, -1*self.Ts]]), [0.0], reals, sense="E")
        solver.add_conv_constraint(dynamics_constraints) 



    def add_input_constraints(self, solver, robot):
        # To minimize over L1 norm of inputs, we minimize over the auxiliary variables and bound the input from above
        # and below with the auxiliary variables
        solver.convex_solver.objective.set_linear(solver.reals[robot['ux_norm']], 1.0)
        solver.convex_solver.objective.set_linear(solver.reals[robot['uy_norm']], 1.0)

        reals = [solver.reals[robot['ux']], solver.reals[robot['ux_norm']]]
        # ux \le uxNorm   <==>  ux - uxNorm \le 0
        L1_norm_constraints = LPClause(np.array([[1.0, -1.0]]), [0.0], reals, sense="L")
        solver.add_conv_constraint(L1_norm_constraints)
        # ux >= -1 * uxNorm  <==>  -uxNorm - ux \le 0
        L1_norm_constraints = LPClause(np.array([[-1.0, -1.0]]), [0.0], reals, sense="L")
        solver.add_conv_constraint(L1_norm_constraints)

        reals = [solver.reals[robot['uy']], solver.reals[robot['uy_norm']]]
        # uy \le uyNorm   <==>  uy - uyNorm \le 0
        L1_norm_constraints = LPClause(np.array([[1.0, -1.0]]), [0.0], reals, sense="L")
        solver.add_conv_constraint(L1_norm_constraints)
        # uy >= -1 * uyNorm  <==>  -uyNorm - uy \le 0
        L1_norm_constraints = LPClause(np.array([[-1.0, -1.0]]), [0.0], reals, sense="L")
        solver.add_conv_constraint(L1_norm_constraints)


        # max limit on ux
        input_constraint = LPClause(np.array([[1.0]]), [self.input_constraints['ux_max']], [solver.reals[robot['ux']]], sense='L')
        solver.add_conv_constraint(input_constraint)

        # min limit on ux
        input_constraint = LPClause(np.array([[1.0]]), [self.input_constraints['ux_min']], [solver.reals[robot['ux']]], sense='G')
        solver.add_conv_constraint(input_constraint)

        # max limit on uy
        input_constraint = LPClause(np.array([[1.0]]), [self.input_constraints['uy_max']], [solver.reals[robot['uy']]], sense='L')
        solver.add_conv_constraint(input_constraint)

        # min limit on uy
        input_constraint = LPClause(np.array([[1.0]]), [self.input_constraints['uy_min']], [solver.reals[robot['uy']]], sense='G')
        solver.add_conv_constraint(input_constraint)
