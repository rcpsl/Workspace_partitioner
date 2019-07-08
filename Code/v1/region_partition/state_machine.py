from __future__ import print_function
import multiprocessing as mp
import timeit
from Workspace import *
from abstract_refinement import *
# from VerifyNNParser import *
from constant import *
import pickle
import argparse


def vertices_to_polyhedron(vertices):
    vertices_cdd = []
    for vertex in vertices:
        vertices_cdd.append([1, vertex[0], vertex[1]])
    # Convert by pycddlib
    mat = cdd.Matrix(vertices_cdd, number_type='float')
    poly = cdd.Polyhedron(mat)
    ine = poly.get_inequalities()
    # Represent inequality constraints as A x <= b
    A, b = [], []
    for row in ine:
        b.append(row[0])
        a = [-x for x in list(row[1:])]
        A.append(a)

    return A,b
def offline_preparation(abst_refinement):
    # Initialize workspace
    workspace = Workspace(num_vertices,num_refined_lasers,'obstacles.json')
    events, segments, event_queue = workspace.prepare_workspace()
    #print('Number of endpoints: ', len(events))
    print('Number of segments: ', len(segments))

    events, segments = sweep_intersections(events, segments, event_queue)
    print('Number of events, ', len(events))
    #for event in events:
    #    print(event)
    #for segment in segments:
    #    print(segment)     

    refined_reg_V_rep_dict, refined_reg_H_rep_dict, lidar_config_dict = None, None, None

    # When measure time of partitioning workspace, turn abstract refinement off
    if abst_refinement == 'ON':
        abst_reg_V_rep, abst_reg_H_rep, refined_reg_V_rep_dict, refined_reg_H_rep_dict, lidar_config_dict \
            = abstract_refinement_partition(workspace, events, segments)
    else: 
        abst_reg_V_rep, abst_reg_H_rep = partition_regions(events, segments)
        lidar_config_dict = workspace.find_lidar_configuration(abst_reg_V_rep)

    #Add obstacles to regions
    obstacles = workspace.obstacles
    for obs in obstacles:
        abst_reg_V_rep.append([(v[0],v[1]) for v in obs])
        A,b = vertices_to_polyhedron(obs)
        abst_reg_H_rep.append({'A':A,'b':b})

    num_abst_reg = len(abst_reg_V_rep)
    print('Number of abstract regions = ', num_abst_reg)
    if(out_file and len(out_file) > 0):
        f = open(out_file,'a+')
        f.write('\t   %6d'%num_abst_reg)
        f.close()

    return abst_reg_V_rep, abst_reg_H_rep, refined_reg_V_rep_dict, refined_reg_H_rep_dict, lidar_config_dict



def check_transitions_given_frm_abst_reg(frm_abst_reg_index, frm_abst_reg_H, abst_reg_H_rep, refined_reg_H_rep_dict, lidar_config_dict, abst_reg_obstacles, parser):
    """
    Check feasibility of transitions start from refined regions in a given abstract region, 
    to obstacles and all other refined regions in workspace
    """
    count_transition_checks_per_core = 0
    for to_abst_reg_index, to_abst_reg_H in enumerate(abst_reg_H_rep):
        print('\n========== End Abstract Region Index = ', to_abst_reg_index, '==========')

        # Branch on two cases: start and end abstract regions are same or not
        if frm_abst_reg_index != to_abst_reg_index:

            # Iterate over refined regions in current start abstract region
            for frm_refined_reg_index, frm_refined_reg_H in enumerate(refined_reg_H_rep_dict[frm_abst_reg_index]):
                frm_lidar_config = lidar_config_dict[frm_abst_reg_index][frm_refined_reg_index]

                # Abstract refinement: check refined regions as end only if abstract region as end is feasible
                abst_solution = parser.parse(frm_refined_reg_H, to_abst_reg_H, frm_lidar_config)
                count_transition_checks_per_core += 1

                if abst_solution:
                    # TODO: Decide which refined region contains above solution by solving abstract solution as end.

                    # Iterate over refined regions in current end abstract region
                    for to_refined_reg_index, to_refined_reg_H in enumerate(refined_reg_H_rep_dict[to_abst_reg_index]):
                        # TODO: No need to check feasibility for end refined region if it is above abtract solution
                        refined_solution = parser.parse(frm_refined_reg_H, to_refined_reg_H, frm_lidar_config)
                        count_transition_checks_per_core += 1

        else:
            # Check transition feasibility between refined regions in the same abstract region
            for frm_refined_reg_index, frm_refined_reg_H in enumerate(refined_reg_H_rep_dict[frm_abst_reg_index]):
                frm_lidar_config = lidar_config_dict[frm_abst_reg_index][frm_refined_reg_index]

                for to_refined_reg_index, to_refined_reg_H in enumerate(refined_reg_H_rep_dict[to_abst_reg_index]):
                    # Start and end should be different refined regions
                    if frm_refined_reg_index != to_refined_reg_index:
                        refined_solution = parser.parse(frm_refined_reg_H, to_refined_reg_H, frm_lidar_config)
                        count_transition_checks_per_core += 1


    # Obstacles 
    for frm_refined_reg_index, frm_refined_reg_H in enumerate(refined_reg_H_rep_dict[frm_abst_reg_index]):
        frm_lidar_config = lidar_config_dict[frm_abst_reg_index][frm_refined_reg_index]

        for obstacle in abst_reg_obstacles:
            print('\n========== Obstacle =========== ')
            obstacle_solution = parser.parse(frm_refined_reg_H, obstacle, frm_lidar_config)
            count_transition_checks_per_core += 1

    return count_transition_checks_per_core



def build_state_machine(num_cores):
    # TODO: Store and load offline results
    start_time = timeit.default_timer()
    abst_reg_V_rep, abst_reg_H_rep, refined_reg_V_rep_dict, refined_reg_H_rep_dict, lidar_config_dict = offline_preparation(abst_refinement='OFF')
    end_time   = timeit.default_timer()
    diff = end_time - start_time
    print('Partition workspace time = ', diff)
    if(out_file and len(out_file) > 0):
            f = open(out_file,'a+')
            f.write('\t%.4f'%diff)
            f.close()
    # Initialize workspace
    #workspace = Workspace()
    
    # Initialize trained NN
    #trained_nn = NeuralNetwork()

    # Initialize VerifyNNParser
    #parser = VerifyNNParser(workspace, trained_nn, num_integrators, Ts, input_limit)


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

    outputFileName = 'results/abst_reg_H_rep.txt'
    with open(outputFileName, 'wb') as outputFile:
        pickle.dump(abst_reg_H_rep, outputFile)
    outputFile.close()

    outputFileName = 'results/abst_reg_V_rep.txt'
    with open(outputFileName, 'wb') as outputFile:
        pickle.dump(abst_reg_V_rep, outputFile)
    outputFile.close()

    outputFileName = 'results/refined_reg_H_rep_dict.txt'
    with open(outputFileName, 'wb') as outputFile:
        pickle.dump(refined_reg_H_rep_dict, outputFile)
    outputFile.close()

    outputFileName = 'results/refined_reg_V_rep_dict.txt'
    with open(outputFileName, 'wb') as outputFile:
        pickle.dump(refined_reg_V_rep_dict, outputFile)
    outputFile.close()

    outputFileName = 'results/lidar_config_dict.txt'
    with open(outputFileName, 'wb') as outputFile:
        pickle.dump(lidar_config_dict, outputFile)
    outputFile.close()


    # TODO: Is the order of regions guaranteed to be same between each run of program? Note used set() in sweep.
    # TODO: When solution is found, add connection to state machine, which is a data structure similar to Graph.
    # TODO: Each process return a list of feasible transitions.    
    """
    start_time = timeit.default_timer()
    pool       = mp.Pool(processes=num_cores)
    results    = [pool.apply_async(check_transitions_given_frm_abst_reg,  
                    args=(i, x, abst_reg_H_rep, refined_reg_H_rep_dict, lidar_config_dict,
                    workspace.abst_reg_obstacles, parser)) for i, x in enumerate(abst_reg_H_rep)]
    output     = [p.get() for p in results]
    end_time   = timeit.default_timer()

    count_transition_checks = sum(output)
    print('\n\nNumber of transitions from each abstract region = ', output)
    print('Total number of transition checks = ', count_transition_checks)
    print('Verification time = ', end_time - start_time)
    """

def create_cmd_parser():
    arg_parser = argparse.ArgumentParser(description='Input arguments)')
    arg_parser.add_argument('num_vertices', help ="Number of vertices")
    arg_parser.add_argument('num_lasers', help ="Number of laser")
    arg_parser.add_argument('--file', help ="Output file")
    return arg_parser


if __name__ == '__main__':
    #np.random.seed(0)
    arg_parser = create_cmd_parser()
    ns = arg_parser.parse_args()
    num_vertices = int(ns.num_vertices)
    num_refined_lasers = int(ns.num_lasers)
    out_file = ns.file
    build_state_machine(num_cores=4)



    
    