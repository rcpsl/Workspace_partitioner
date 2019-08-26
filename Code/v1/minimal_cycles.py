from __future__ import print_function
from operator import itemgetter
import cdd
from Graph import *
from sweep import *


def build_graphs(events, segments, num_abst_reg=0, abst_reg_V_rep=[]):
    """
    Build one graph if num_abst_reg is zero, otherwise one graph for each abstract region
    """
    if num_abst_reg:
        graphs = {}
        # NOTE: Key for graphs are index of abstract regions in abst_reg_V_rep
        for abst_reg_index, this_abst_reg in enumerate(abst_reg_V_rep):
            graphs[abst_reg_index] = Graph()
    else:    
        graph = Graph()
    #for segment in segments:
    #    segment.parent, segment.right, segment.left = None, None, None

    # NOTE: This function is called multiple times. Must clean attributes
    for segment in segments:
        segment.upper, segment.lower, segment.interior = None, None, []

    for event in events:
        #event.parent, event.right, event.left = None, None, None
        if num_abst_reg:
            for abst_reg_index in event.abst_reg:
                graphs[abst_reg_index].add_vertex((event.x, event.y))
        else:
            graph.add_vertex((event.x, event.y))

        for segment in event.U:
            segment.upper = event
        for segment in event.L:
            segment.lower = event
        for segment in event.C:
            segment.interior.append(event)

   
    for segment in segments:
        # Sort events on each segment in ascending order, where the total order is same as defined in Event class.
        # Sorting by keys is expected to be faster than sort()
        segment.interior.sort(key=lambda event: event.x)
        segment.interior.sort(key=lambda event: event.y, reverse=True)
        events_on_seg = segment.interior
        #NOTE: Modification on event_on_seg also changes segment.interior
        events_on_seg.insert(0, segment.upper)
        events_on_seg.append(segment.lower)
        
        # A pair of directed edges between neighbor events on a segment have opposite directions
        for i in xrange(len(events_on_seg)-1):
            small_event = events_on_seg[i]
            big_event = events_on_seg[i+1]
            small_event_coordinates = (small_event.x, small_event.y)
            big_event_coordinates   = (big_event.x, big_event.y)
    
            # TODO: It is inefficient to check this branch in nested loops
            if num_abst_reg:
                # Find atract regions contain both events
                abst_reg_shared = small_event.abst_reg.intersection(big_event.abst_reg)
                assert len(abst_reg_shared) <= 2, 'A part of segment can be share by at most two abstract regions'
                for abst_reg_index in abst_reg_shared:
                    # Principal lasers become boundary
                    is_boundary = True if segment.is_principal else False
                    graphs[abst_reg_index].add_edge(small_event_coordinates, big_event_coordinates, {'orientation': segment.angle+180.0, 'is_boundary': is_boundary, 'visited': False})
                    graphs[abst_reg_index].add_edge(big_event_coordinates, small_event_coordinates, {'orientation': segment.angle, 'is_boundary': is_boundary, 'visited': False})
            else:
                graph.add_edge(small_event_coordinates, big_event_coordinates, {'orientation': segment.angle+180.0, 'is_boundary': segment.is_boundary, 'visited': False})
                graph.add_edge(big_event_coordinates, small_event_coordinates, {'orientation': segment.angle, 'is_boundary': segment.is_boundary, 'visited': False})

                #if segment.name == 's8':
                #    for i in xrange(len(events_on_seg)-1):
                #        this_event = events_on_seg[i]
                #        next_event = events_on_seg[i+1]
                #        print('this event', (this_event.x, this_event.y), 'next event', (next_event.x, next_event.y))
                #    print('\n')    

        """ 
        for v in graph:
            for w in v.get_connections():
                vid = v.id()
                wid = w.id()
                print(vid, wid, v.adjacent[w]['is_boundary'])
        """

        #for v in graph:
        #    print('graph.vertex_dict[%s]=%s' % (v.get_id(), graph.vertex_dict[v.get_id()]))
    if num_abst_reg:
        return graphs
    else:
        return graph



def rightmost_search(graph, start_vertex, v, w, cycle):
    # Find the rightmost outgoing edge from w relative to incoming edge (v, w)
    alpha = v.adjacent[w]['orientation'] # Direction of incoming edge
    rightmost_angle = 360.0

    #print(w.id)
    for p in w.get_connections():
        # The rightmost edge should always have not been visited yet
        # NOTE: This if statment avoids unnecessary comparisons, but also makes following assert on rightmost edge is unvisited trivial 
        if not w.adjacent[p]['visited']:
            beta = w.adjacent[p]['orientation'] # Direction of outgoing edge
            #print('v =', v.id, ', w=', w.id, ', p=', p.id, ', alpha =', alpha, ', beta=', beta)

            # TODO: If angles are not precise, need to consider round-off error beyond subtraction and addition
            if alpha >= 180.0 and alpha-180.0+abs_tol <  beta < alpha:
                if beta < rightmost_angle:
                    rightmost_angle  = beta
                    rightmost_vertex = p

            elif alpha < 180.0 and (alpha+180.0+abs_tol < beta <= 360.0 or 0.0 < beta < alpha): 
                if alpha+180.0+abs_tol < beta <= 360.0:
                    aligned_beta = beta - 360.0
                else:
                    aligned_beta = beta
                if aligned_beta < rightmost_angle: # Not need abs_tol since each angle is processed only once
                    rightmost_angle  = aligned_beta
                    rightmost_vertex = p
                
    assert rightmost_angle != 360.0, 'A right edge with turning less than 180 degrees does not exist'
    # TODO: Turn off above if statement to make following assert non-trivial
    #assert not w.adjacent[rightmost_vertex]['visited'], 'The rightmost edge should have not been visited'

    w.adjacent[rightmost_vertex]['visited'] = True
    cycle.append(rightmost_vertex.id)  

    if not is_close(rightmost_vertex.id[0], start_vertex.id[0]) or not is_close(rightmost_vertex.id[1], start_vertex.id[1]):
        rightmost_search(graph, start_vertex, w, rightmost_vertex, cycle)

      

def detect_minimum_cycles(graph):
    """
    Return both V and H representation of polyhedra correspond to minimal cycles
    """
    minimal_cycles = []
    for v in graph:
        for w in v.get_connections():
            # The first step should not be a boundary edge
            if not v.adjacent[w]['is_boundary'] and not v.adjacent[w]['visited']:
                #print('\n')
                #print(v)
                v.adjacent[w]['visited'] = True
                cycle = [v.id, w.id]
                rightmost_search(graph, v, v, w, cycle)
                cycle = cycle[:-1]
                minimal_cycles.append(cycle)
    
    #print('\nNumber of minimal cycles = ', len(minimal_cycles))
    #for cycle in minimal_cycles:
    #    print(cycle, end='\n\n')
    #print(minimal_cycles)

    # Sanity check
    #print('\nSanity check all unvisited edges: ')
    num_unvisited_edges = 0
    for v in graph:
        for w in v.get_connections():
            if v.adjacent[w]['visited'] == False:
                num_unvisited_edges += 1
                vid, wid = v.id, w.id
                #print(vid, '-->', wid)
                assert v.adjacent[w]['is_boundary'] == True, 'Unvisited edge must be a boundary edge'
                # The following assert is activated only when abstract regtion is partioned by refined lasers
                if graph.num_vertices != graph.num_edges/2: 
                    assert w.adjacent[v]['visited'] == True, 'Counterpart of an unvisited boundary edge must be visited'
                # TODO: Check unvisted edges cover workspace and obstacle boundaries
    #print('Number of unvisited edges = ', num_unvisited_edges)

    # Convert V-reprsentation of polyhedra to H-Reprentation
    # NOTE: Region orders are same in V and H representations
    poly_H_rep = []
    for cycle in minimal_cycles:
        # V-representation required by pycddlib
        vertices = []
        for vertex in cycle:
            vertices.append([1, vertex[0], vertex[1]])
        #print(vertices, end='\n\n')

        # Convert by pycddlib
        mat = cdd.Matrix(vertices, number_type='float')
        poly = cdd.Polyhedron(mat)
        ine = poly.get_inequalities()
        # TODO: need canonicalize() to remove redundancy?
        #ine.canonicalize()

        # Represent inequality constraints as A x <= b
        A, b = [], []
        for row in ine:
            b.append(row[0])
            a = [-x for x in list(row[1:])]
            A.append(a)
        #print('b = ', b)     
        #print('A = ', A, end='\n\n')

        poly_H_rep.append({'A': A, 'b': b})
             
    #print(poly_H_rep)

    return minimal_cycles, poly_H_rep



def partition_regions(events, segments, num_abst_reg=0, abst_reg_V_rep=[]):
    """
    Return both V and H representation of polyhedra correspond to minimal cycles.
    Find minimal cycles in each abstract region seperately if num_abst_reg is non-zero
    """
    if num_abst_reg:
        graphs = build_graphs(events, segments, num_abst_reg=num_abst_reg, abst_reg_V_rep=abst_reg_V_rep)
        # Store V and H representation of refined regions in each abstract region separately
        minimal_cycles_dict, poly_H_rep_dict = {}, {}
        for abst_reg_index in xrange(num_abst_reg):
            graph = graphs[abst_reg_index]
            minimal_cycles, poly_H_rep          = detect_minimum_cycles(graph)
            minimal_cycles_dict[abst_reg_index] = minimal_cycles
            poly_H_rep_dict[abst_reg_index]     = poly_H_rep
        return minimal_cycles_dict, poly_H_rep_dict

    else:     
        graph = build_graphs(events, segments)
        #print('Total number of vertices ', graph.num_vertices)
        #print('Total number of edges ', graph.num_edges) 
        minimal_cycles, poly_H_rep = detect_minimum_cycles(graph)
        return minimal_cycles, poly_H_rep
    