from __future__ import print_function
import matplotlib.path as mpltPath
import cdd
from minimal_cycles import *


def inflate_abst_reg(reg_H):
    """
    Input H-rep of a polyhedron, inflated by Ax <= b + abs_tol * 1 (where 1 is a vector of all ones).
    Return V-rep of inflated polyhedron
    """
    A, b = reg_H['A'], reg_H['b']
    # ine is in form [b -A]
    ine = []
    for i, param in enumerate(b):
        param += abs_tol
        row = [param] + [-x for x in A[i]]
        ine.append(row)

    mat = cdd.Matrix(ine, number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    # Convert H-rep to V-rep
    ext = poly.get_generators()
    poly_inflated = [(vertex[1], vertex[2]) for vertex in ext]
    return poly_inflated



def abstract_refinement_partition(workspace, events, segments):
    """
    Partition workspace into abstract and refined regions
    """
    principal_events   = [x for x in events if x.is_principal]
    principal_segments = [x for x in segments if x.is_principal]
    print('Number of principal segments = ', len(principal_segments))  
    print('Number of principal events = ', len(principal_events))
     

    # NOTE: Abstract regions have the same order in abst_reg_V_rep and abst_reg_H_rep
    abst_reg_V_rep, abst_reg_H_rep = partition_regions(principal_events, principal_segments)
    num_abst_reg = len(abst_reg_V_rep)
    print('Number of abstract regions = ', num_abst_reg)
    #print('Number of abstract regions (in V-rep) = ', len(abst_reg_V_rep))
    #print('Number of abstract regions (in H-rep) = ', len(abst_reg_H_rep))


    # For each event, store a list of indices of abstract regions that the event is contained (include on boundary).
    # TODO: Abstract regions contrain PRINCIPAL events should be known in above partition of workspace into abstract regions
    event_coordinates = [(event.x, event.y) for event in events]

    # Find which events are in each abstract region     
    # TODO: Any better way other than checking containing relations between all events with all abstract regions?
    for event in events:
        event.abst_reg = set()
    for abst_reg_index, this_abst_reg_H in enumerate(abst_reg_H_rep):
        # Inflate abstract region due to float round-off error
        abst_reg_V_inflated = inflate_abst_reg(this_abst_reg_H)
        # Use Matplotlit to check if points are in absract region
        path                 = mpltPath.Path(abst_reg_V_inflated)
        contain_relation     = path.contains_points(event_coordinates)
        inside_event_indices = [i for i, x in enumerate(contain_relation) if x == True]
        # NOTE: Events order is same in event_coordinates and events
        for i in inside_event_indices:
            events[i].abst_reg.add(abst_reg_index)

        # Print some points in each abstract region
        """
        this_abst_reg_V = abst_reg_V_rep[abst_reg_index]
        for i in xrange(len(this_abst_reg_V)):
            print('{:04.2f}, {:04.2f}'.format(this_abst_reg_V[i][0], this_abst_reg_V[i][1]), end=' -> ')
        print('')
        events_inside = [events[i] for i in inside_event_indices]
        for event in events_inside:
            print(event)
        print('\n')
        """

    # NOTE: Key for the following dictionaries is index of abtract regions in abst_reg_V_rep.
    # Each element is a list of refined regions (same order for H and V representations) in the abstract region 
    refined_reg_V_rep_dict, refined_reg_H_rep_dict = partition_regions(events, segments, num_abst_reg=num_abst_reg, abst_reg_V_rep=abst_reg_V_rep) 
    #print('Length of refined_reg_V_rep_dict = ', len(refined_reg_V_rep_dict,))

    # If an abstract region does not contain any refined region, set refined region to be itself
    for abst_reg_index, this_abst_reg_V in enumerate(abst_reg_V_rep):
        if not refined_reg_V_rep_dict[abst_reg_index]:
            refined_reg_V_rep_dict[abst_reg_index] = [this_abst_reg_V]
            refined_reg_H_rep_dict[abst_reg_index] = [abst_reg_H_rep[abst_reg_index]]


    # Find LiDAR configuration for each refined region.
    lidar_config_dict = {}
    # NOTE: Key for lidar_config_dict is index of abstract regions in abst_reg_V_rep
    # NOTE: Each element in lidar_config_dict is a list of LiDAR configurations in the same order as refined regions in this abtract region
    for abst_reg_index, this_abst_reg_V in enumerate(abst_reg_V_rep):
        refined_reg_V_rep_in_this_abst = refined_reg_V_rep_dict[abst_reg_index]
        lidar_config_dict[abst_reg_index] = workspace.find_lidar_configuration(refined_reg_V_rep_in_this_abst)
    #print('Length of lidar_config_dict = ', len(lidar_config_dict))


    # Print refined regions in each abstract region
    """
    for abst_reg_index, this_abst_reg_V in enumerate(abst_reg_V_rep):
        refined_reg_V_rep_in_this_abst = refined_reg_V_rep_dict[abst_reg_index] 
        print('\n----------------------------------------')
        print_region(this_abst_reg_V)
        print('\nNumber of refined region = ', len(refined_reg_V_rep_in_this_abst), '\n')
            
        for refined_reg_index, this_refined_reg_V in enumerate(refined_reg_V_rep_in_this_abst):
            print_region(this_refined_reg_V)
            print('\n')
    """

    return abst_reg_V_rep, abst_reg_H_rep, refined_reg_V_rep_dict, refined_reg_H_rep_dict, lidar_config_dict