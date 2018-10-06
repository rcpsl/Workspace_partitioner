from __future__ import print_function
from Workspace import *
from abstract_refinement import *
from VerifyNNParser import *
from constant import *


def offline_preparation():
    # Initialize workspace
    workspace = Workspace()
    events, segments, event_queue = workspace.prepare_workspace()
    #print('Number of endpoints: ', len(events))
    print('Number of segments: ', len(segments))

    events, segments = sweep_intersections(events, segments, event_queue)
    print('Number of events, ', len(events))

    abst_reg_V_rep, abst_reg_H_rep, refined_reg_V_rep_dict, refined_reg_H_rep_dict, lidar_config_dict \
        = abstract_refinement_partition(workspace, events, segments)

    return abst_reg_V_rep, abst_reg_H_rep, refined_reg_V_rep_dict, refined_reg_H_rep_dict, lidar_config_dict



def build_state_machine():
    # TODO: Add regions for obstacle and workspace boundary.
    # TODO: Store and load offline results
    abst_reg_V_rep, abst_reg_H_rep, refined_reg_V_rep_dict, refined_reg_H_rep_dict, lidar_config_dict = offline_preparation()
    
    # Initialize workspace
    workspace = Workspace()
    
    # Initialize trained NN
    trained_nn = NeuralNetwork()

    # Initialize VerifyNNParser
    parser = VerifyNNParser(workspace, trained_nn, num_integrators, Ts, input_limit)


    # A test by using specified start and end regions
    """
    frm_abst_reg_index    = 15
    frm_refined_reg_index = 3
    to_abst_reg_index     = 4
    to_refined_reg_index  = 1

    frm_abst_reg_V    = abst_reg_V_rep[frm_abst_reg_index]
    frm_abst_reg_H    = abst_reg_H_rep[frm_abst_reg_index]
    frm_refined_reg_V = refined_reg_V_rep_dict[frm_abst_reg_index][frm_refined_reg_index]
    frm_refined_reg_H = refined_reg_H_rep_dict[frm_abst_reg_index][frm_refined_reg_index]
    frm_lidar_config  = lidar_config_dict[frm_abst_reg_index][frm_refined_reg_index]

    to_abst_reg_V     = abst_reg_V_rep[to_abst_reg_index]
    to_abst_reg_H     = abst_reg_H_rep[to_abst_reg_index]
    to_refined_reg_V  = refined_reg_V_rep_dict[to_abst_reg_index][to_refined_reg_index]
    to_refined_reg_H  = refined_reg_H_rep_dict[to_abst_reg_index][to_refined_reg_index]
    to_lidar_config   = lidar_config_dict[to_abst_reg_index][to_refined_reg_index]

    #print(frm_abst_reg_V)
    #print(frm_refined_reg_V)
    #print(frm_lidar_config)
    #print(to_abst_reg_V)
    #print(to_refined_reg_V)
    #print(to_lidar_config)

    frm_poly_H_rep   = frm_refined_reg_H
    to_poly_H_rep    = to_refined_reg_H
    frm_lidar_config = frm_lidar_config

    parser.parse(frm_poly_H_rep, to_poly_H_rep, frm_lidar_config)
    """


    # TODO: When solution is found, add connection to state machine, which is a data structure similar to Graph.
    # TODO: Is the order of regions guaranteed to be same between each run of program? Note used set() in sweep.
    # TODO: Parallelization
    count_transition_checks = 0

    # Iterate over abstract regions as start
    for frm_abst_reg_index, frm_abst_reg_H in enumerate(abst_reg_H_rep):
        print('\n\n**************** Start Abstract Region Index = ', frm_abst_reg_index, '***********************')

        # Iterate over abstract regions as end
        for to_abst_reg_index, to_abst_reg_H in enumerate(abst_reg_H_rep):
            print('\n========== End Abstract Region Index = ', to_abst_reg_index, '==========')

            # Branch on two cases: start and end abstract regions are same or not
            if frm_abst_reg_index != to_abst_reg_index:

                # Iterate over refined regions in current start abstract region
                for frm_refined_reg_index, frm_refined_reg_H in enumerate(refined_reg_H_rep_dict[frm_abst_reg_index]):
                    frm_lidar_config = lidar_config_dict[frm_abst_reg_index][frm_refined_reg_index]

                    # Abstract refinement: check refined regions as end only if abstract region as end is feasible
                    abst_solution = parser.parse(frm_refined_reg_H, to_abst_reg_H, frm_lidar_config)
                    count_transition_checks += 1

                    if abst_solution:
                        # TODO: Decide which refined region contains above solution by solving abstract solution as end.

                        # Iterate over refined regions in current end abstract region
                        for to_refined_reg_index, to_refined_reg_H in enumerate(refined_reg_H_rep_dict[to_abst_reg_index]):
                            # TODO: No need to check feasibility for end refined region if it is above abtract solution
                            refined_solution = parser.parse(frm_refined_reg_H, to_refined_reg_H, frm_lidar_config)
                            count_transition_checks += 1

            else:
                # Check transition feasibility between refined regions in the same abstract region
                for frm_refined_reg_index, frm_refined_reg_H in enumerate(refined_reg_H_rep_dict[frm_abst_reg_index]):
                    for to_refined_reg_index, to_refined_reg_H in enumerate(refined_reg_H_rep_dict[to_abst_reg_index]):
                        # Start and end should be different refined regions
                        if frm_refined_reg_index != to_refined_reg_index:
                            frm_lidar_config = lidar_config_dict[frm_abst_reg_index][frm_refined_reg_index]
                            refined_solution = parser.parse(frm_refined_reg_H, to_refined_reg_H, frm_lidar_config)
                            count_transition_checks += 1

    print('\n\nTotal number of transition checks = ', count_transition_checks)



if __name__ == '__main__':
    np.random.seed(0)
    build_state_machine()



    
    