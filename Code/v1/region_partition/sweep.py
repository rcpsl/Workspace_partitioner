from __future__ import print_function
from BinarySearchTree import *


def find_new_event(event_queue, s, t, event, events, verbose):
    #if verbose == 'ON':
    #    print('Test intersection between ', s.name, ' and ', t.name)

    # Find intersection between s (endpoints a,b) and t (endpoints c,d).
    # NOTE: This has been moved to utility as a routine
    aX, aY, bX, bY = s.x1, s.y1, s.x2, s.y2
    cX, cY, dX, dY = t.x1, t.y1, t.x2, t.y2

    A = (bX - aX) * (cY - dY) - (cX - dX) * (bY - aY)
    if not is_close(A, 0.0):
        As = (cX - aX) * (cY - dY) - (cX - dX) * (cY - aY)
        At = (bX - aX) * (cY - aY) - (cX - aX) * (bY - aY)
        ks = As / A
        kt = At / A
    else: 
        return

    if 0.0-abs_tol <= ks <= 1.0+abs_tol and 0.0-abs_tol <= kt <= 1.0+abs_tol:
        intersection_X = (1 - ks) * aX + ks * bX
        intersection_Y = (1 - ks) * aY + ks * bY
    else: 
        return   
     

    # Interested events are below or on right when have same y-coordinate with current event
    if intersection_Y + abs_tol < event.y or is_close(intersection_Y, event.y) and intersection_X > event.x + abs_tol:
        if verbose == 'ON':    
            print('Found an intersection ', (intersection_X, intersection_Y))    

        # Check if this event is endpoint of both s and t     
        if (is_close(ks, 0.0) or is_close(ks, 1.0)) and (is_close(kt, 0.0) or is_close(kt, 1.0)):
            #if verbose == 'ON':
            #    print('This intersection is endpoint of both segments')
            return

        e = Event(intersection_X, intersection_Y, set(), set(), set(), is_principal=False)
        # This event may have been detected earlier.
        # TODO: Search and insert by just walking down BST once
        new_event = event_queue.search(event_queue.root, e)
        if new_event is None:
            new_event = e
            event_queue.insert(new_event)
            events.append(new_event)
        #else:
            #if verbose == 'ON':
            #    print('This intersection is already in event queue')
        if not new_event.is_principal and s.is_principal and t.is_principal:
            new_event.is_principal = True

        # The new event is not endpoints of s
        if not is_close(ks, 0.0) and not is_close(ks, 1.0):
            new_event.C.add(s)    
        # The new event is not endpoints of t
        if not is_close(kt, 0.0) and not is_close(kt, 1.0):     
            new_event.C.add(t)        
    
    #if verbose == 'ON':
    #    print('')



def handle_event(event_queue, sweep_status, event, events, verbose='OFF'):
    """
    if verbose == 'ON':
        print('\n=====================================')
        print('Processing event ', (event.x, event.y))
        print('=====================================')
        print('U = ', end='')
        for segment in event.U:
            print(segment.name, end=', ')
        print('\nC = ', end='')
        for segment in event.C:
            print(segment.name, end=',')
        print('\nL = ', end='')
        for segment in event.L:
            print(segment.name, end=',')
        print('\n') 
    """
    # Consider two cases: U union C of this event is empty or nonempty.
    # Note that it is possible that C does not yet include all segments that contain this event in their interior.
    # This may only happen when C is empty but U is nonempy, and this will be handled under the U union C is nonempty case
    U_C_union = event.U.union(event.C)

    if not bool(U_C_union): # empty
        # Find the leftmost and rightmost segment in L of this event, where the order is current sweep status 
        # (before updating intersections with sweep line).
        # This order corresponds to segments in sweep status that are intersected by a sweep line just ABOVE this event. 
        segment = next(iter(event.L))
        leftmost_in_L, rightmost_in_L = segment, segment
        for segment in event.L:
            if segment < leftmost_in_L:
                leftmost_in_L = segment
            if rightmost_in_L < segment:     
                rightmost_in_L = segment      

        # Left and right neighbor in sweep status of the leftmost and rightmost segment, respectively
        left_neighbor  = sweep_status.predecessor(leftmost_in_L)
        right_neighbor = sweep_status.successor(rightmost_in_L)

        if left_neighbor is not None and right_neighbor is not None:
            find_new_event(event_queue, left_neighbor, right_neighbor, event, events, verbose)

        for segment in event.L:
            sweep_status.delete(segment)
            segment.parent, segment.left, segment.right = None, None, None        
        # Update x-coordinate of intersections between segments with sweep line.
        # Since U union C is empty, this update does not change order of segments in sweep status.
        # TODO: Is this update necessary?     
        sweep_status.update_sweep_intersections(sweep_status.root, event)  

        #if verbose == 'ON':
        #    print('sweep status after updating:')
        #    sweep_status.inorder_walk(sweep_status.root)
        #    print('\n')


    else: # U union C is nonempty
        # Delete segments whose lower endpoint is this event,
        # insert segments whose upper endpoint is this event,
        # delete and re-insert segments contain this event to reverse the order
        for segment in event.L.union(event.C):
            sweep_status.delete(segment)
            segment.parent, segment.left, segment.right = None, None, None

        # Sweep status keeps same even after updating keys, since at most one segment (the one should
        # be in C but undetected yet) that starts or contains this event remains in sweep status after above deletion
        sweep_status.update_sweep_intersections(sweep_status.root, event)   
    
        for segment in U_C_union:
            segment.evaluate_sweep_intersection(event)
            # TODO: Randomize insertion order to keep BST be approximately balanced.
            # No need to search before insertion due to above deletion
            sweep_status.insert(segment)

        #if verbose == 'ON':
        #    print('sweep status after updating:')
        #    sweep_status.inorder_walk(sweep_status.root)
        #    print('\n')

        # As noted at the begining of this function, it is necessary to check if there are
        # other segments that contain this event in their interior but are not in C yet. 
        # If there is, there should be only one such segment, and should be a neighbor of segments of U in sweep status
        if not bool(event.C): 
            start_search     = min(event.U, key = lambda segment: segment.angle)
            end_search       = max(event.U, key = lambda segment: segment.angle)
            left_neighbor_U  = sweep_status.predecessor(start_search)
            right_neighbor_U = sweep_status.successor(end_search)

            if left_neighbor_U is not None:
                start_search = left_neighbor_U
            if right_neighbor_U is not None:
                end_search = right_neighbor_U                   

            segment = start_search
            while 1:
                # Check if segment contains this event in its interior
                if is_between(segment.x1, segment.y1, segment.x2, segment.y2, event.x, event.y):
                    if verbose == 'ON':
                        print((event.x, event.y), 'is an undetected intersection as interior point of segment ', segment.name)
                    event.C.add(segment)
                    U_C_union.add(segment)
                    break
                else:      
                    if segment == end_search:
                        #if verbose == 'ON':
                        #    print('\nNo undetected segments should be in empty C\n')
                        break
                    segment = sweep_status.successor(segment)

        # Find the leftmost and rightmost segment of U union C, where the order is current sweep status
        # (after updating intersections with sweep line).
        # This order corresponds to segments in sweep status that are intersected by a sweep line just BELOW this event. 
        # Since all segments in U union C intersect sweep line at the same point, may just compare angle
        leftmost_in_UC  = min(U_C_union, key = lambda segment: segment.angle)
        rightmost_in_UC = max(U_C_union, key = lambda segment: segment.angle)
        
        # Left and right neighbor in sweep status of the leftmost and rightmost segment, respectively
        left_neighbor  = sweep_status.predecessor(leftmost_in_UC)
        right_neighbor = sweep_status.successor(rightmost_in_UC)

        if left_neighbor is not None:
            find_new_event(event_queue, left_neighbor, leftmost_in_UC, event, events, verbose)
        if right_neighbor is not None:    
            find_new_event(event_queue, right_neighbor, rightmost_in_UC, event, events, verbose)



def sweep_intersections(events, segments, event_queue):
    """
    Find all intersections of given segments
    """
    # TODO: use red-black tree for event queue and sweep line status structure. But is it more efficient
    # than just using list and built-in routines such as sort? 
    #event_queue  = EventQueue()
    sweep_status = SweepStatus()

    # Insert segment endpoints into event queue.
    # TODO: Randomize insertion order to keep BST be approximately balanced.
    # NOTE: If want to characterize segment, can use angle and the first endpoint, which has high precision by using obstacle corners 

    # Test 1
    """
    s0 = Segment(4.0, 1.0, 2.0, 4.0, 123.69, 's0')
    s1 = Segment(2.0, 4.0, 0.0, -2.0, 71.57, 's1')
    s2 = Segment(3.0, -3.0, 2.0, 4.0, 98.13, 's2')
    s3 = Segment(-1.0, 3.0, 1.0, 1.0, 135.0, 's3')
    s4 = Segment(-1.0, 1.0, 3.0, -5.0, 123.69, 's4')
    s5 = Segment(1.0, 1.0, 1.0, -2.0, 90.0, 's5')
    s6 = Segment(4.0, -2.0, 1.0, 1.0, 135.0, 's6')
    s7 = Segment(4.0, 1.0, 1.0, 1.0, 180.0, 's7')
    segments = [s0, s1, s2, s3, s4, s5, s6, s7]

    e0 = Event(2.0, 4.0, set([s0, s1, s2]), set(), set())
    e1 = Event(0.0, -2.0, set(), set([s1]), set())
    e2 = Event(3.0, -3.0, set(), set([s2]), set())
    e3 = Event(1.0, 1.0, set([s7, s6, s5]), set([s3]), set())
    e4 = Event(-1.0, 3.0, set([s3]), set(), set())
    e5 = Event(3.0, -5.0, set(), set([s4]), set())
    e6 = Event(-1.0, 1.0, set([s4]), set(), set())
    e7 = Event(1.0, -2.0, set(), set([s5]), set())
    e8 = Event(4.0, -2.0, set(), set([s6]), set())
    e9 = Event(4.0, 1.0, set(), set([s0, s7]), set())
    events = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]
    """


    # Test 2
    """
    s1 = Segment(-5.0, 2.0, -2.0, 2.0, 180.0, 's1')
    s2 = Segment(-3.0, 2.0, 1.0, -2.0, 135.0, 's2')
    s3 = Segment(0.0, 2.0, 4.0, 2.0, 180.0, 's3')
    s4 = Segment(-5.0, -2.0, 3.0, 2.0, 26.6, 's4')
    s5 = Segment(-6.0, 4.0, -3.0, -2.0, 116.6, 's5')
    s6 = Segment(-3.0, 4.0, -2.0, 2.0, 116.6, 's6')
    s7 = Segment(-2.5, 0.0, 0.0, 5.0, 63.4, 's7')
    s8 = Segment(-2.5, 1.0, -2.5, 2.0, 90.0, 's8')
    s9 = Segment(-1.0, -2.0, 4.0, 2.0, 38.7, 's9')

    e1 = Event(-5.0, 2.0, set([s1]), set(), set())
    e2 = Event(-3.0, 2.0, set([s2]), set(), set())
    e3 = Event(-2.0, 2.0, set(), set([s1, s6]), set())
    e4 = Event(0.0, 2.0, set([s3]), set(), set())
    e5 = Event(3.0, 2.0, set([s4]), set(), set())
    e6 = Event(4.0, 2.0, set([s9]), set([s3]), set())
    e7 = Event(-6.0, 4.0, set([s5]), set(), set())
    e8 = Event(-3.0, 4.0, set([s6]), set(), set())
    e9 = Event(0.0, 5.0, set([s7]), set(), set())
    e11 = Event(-2.5, 0.0, set(), set([s7]), set())
    e14 = Event(-5.0, -2.0, set(), set([s4]), set())
    e15 = Event(-3.0, -2.0, set(), set([s5]), set())
    e16 = Event(1.0, -2.0, set(), set([s2]), set())
    e17 = Event(-2.5, 2.0, set([s8]), set(), set())
    e18 = Event(-2.5, 1.0, set(), set([s8]), set())
    e21 = Event(-1.0, -2.0, set(), set([s9]), set())
    events = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e11, e14, e15, e16, e17, e18, e21]
    """


    # Test 3
    """
    s1 = Segment(3.0, 5.0, 1.0, 7.0, 135.0, 's1')
    s2 = Segment(3.0, 5.0, 7.0, 1.0, 135.0, 's2')
    s3 = Segment(1.0, 1.0, 4.0, 7.0, 63.43, 's3')
    s4 = Segment(4.0, 7.0, 5.0, 1.0, 99.46, 's4')
    s5 = Segment(10.0, 7.0, 1.0, 7.0, 180.0, 's5', is_boundary=True)
    s6 = Segment(1.0, 1.0, 1.0, 7.0, 90.0, 's6', is_boundary=True)
    s7 = Segment(3.0, 1.0, 3.0, 5.0, 90.0, 's7')
    s8 = Segment(8.0, 5.0, 3.0, 5.0, 180.0, 's8')
    s9 = Segment(7.5, 7.0, 6.0, 1.0, 75.96, 's9')
    s10 = Segment(8.0, 1.0, 1.0, 1.0, 180.0, 's10', is_boundary=True)
    s11 = Segment(10.0, 5.0, 10.0, 7.0, 90.0, 's11', is_boundary=True)
    s12 = Segment(8.0, 5.0, 8.0, 1.0, 90.0, 's12', is_boundary=True)
    s13 = Segment(8.0, 5.0, 10.0, 5.0, 180.0, 's13', is_boundary=True)
    s14 = Segment(8.0, 5.0, 8.5, 7.0, 75.93, 's14')

    e1 = Event(3.0, 5.0, set([s2, s7, s8]), set([s1]), set())
    e4 = Event(8.0, 5.0, set([s12, s13]), set([s8, s14]), set())
    e5 = Event(10.0, 5.0, set(), set([s13, s11]), set())
    e6 = Event(1.0, 1.0, set([s10]), set([s3, s6]), set())
    e7 = Event(6.0, 1.0, set(), set([s9]), set())
    e8 = Event(5.0, 1.0, set(), set([s4]), set())
    e9 = Event(3.0, 1.0, set(), set([s7]), set())
    e10 = Event(7.0, 1.0, set(), set([s2]), set())
    e11 = Event(8.0, 1.0, set(), set([s10, s12]), set())
    e14 = Event(10.0, 7.0, set([s11]), set([s5]), set())
    e15 = Event(4.0, 7.0, set([s3, s4]), set(), set())
    e16 = Event(7.5, 7.0, set([s9]), set(), set())
    e17 = Event(1.0, 7.0, set([s1, s5, s6]), set(), set()) 
    e18 = Event(8.5, 7.0, set([s14]), set(), set()) 

    segments = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14]
    events   = [e15, e16, e5, e14, e17, e9, e10, e11, e1, e6, e7, e8, e18, e4]
    """

    #for event in events:
    #    event_queue.insert(event)

    while event_queue.root is not None:
        # Return event that has biggest y-coordinate or smallest x-coordinate when y-coordinates are same
        event = event_queue.minimum(event_queue.root)
        event_queue.delete(event)
        event.parent, event.left, event.right = None, None, None
        handle_event(event_queue, sweep_status, event, events, verbose='OFF')

    return events, segments


