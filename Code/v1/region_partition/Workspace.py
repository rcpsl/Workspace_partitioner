from utility import *
from BinarySearchTree import *
import numpy as np


class Workspace(object):

    def __init__(self):
        """
        Define workspace and LiDAR parameters
        """
        # LiDAR parameters
        self.max_range = 100.0

        # TODO: Currently does not support vertical (90 and 270) and horizontal (0 and 180) lasers due to using tan, cot.
        # NOTE: Laser angles are in range [0, 360) degrees.
        # NOTE: It might be good to define angles in precise. Otherwise, need to consider round-off error in sweep and minimal cycles

        ######################################
        # Use this configuration to generate regions AND experiments for Table 2-4
        ######################################
        #num_vertices = 8
        #self.principal_angles = [67.5, 112.5, 247.5, 292.5]
        #refined_angles = [22.5, 157.5, 202.5, 337.5]
        
        ######################################
        # Use the following configurations for Table 1.
        # Just need to set num_vertices and num_refined_lasers be same as Table 1
        ######################################
        num_vertices = 8
        num_refined_lasers = 32
        self.principal_angles = []
        num_refined_lasers += 2
        refined_angles = np.linspace(0, 360, num_refined_lasers) 
        refined_angles = refined_angles.tolist()
        refined_angles = [x for x in refined_angles if x not in [0.0, 90.0, 180.0, 270.0, 360.0]]
        if num_vertices == 12 and (num_refined_lasers-2 == 218 or num_refined_lasers-2 == 298):
            refined_angles = [math.ceil(x*100)/100 for x in refined_angles] 
        else:      
            refined_angles = [math.ceil(x*10)/10 for x in refined_angles]
        

        ######################################
        # Ignore the following commented code      
        ######################################
        # Not abstract refinement when measure time of partioning regions, so principal_angles is empty
        #self.principal_angles = [82.87, 97.13, 262.87, 277.13]
        #self.principal_angles = [67.5, 112.5, 247.5, 292.5]
        #self.principal_angles = []

        #refined_angles = [22.5, 157.5, 202.5, 337.5]
        #num_refined_lasers = 122
        #refined_angles = np.linspace(0, 360, num_refined_lasers) 
        #refined_angles = refined_angles.tolist()
        #refined_angles = [x for x in refined_angles if x not in [0.0, 90.0, 180.0, 270.0, 360.0]]

        #refined_angles1  = 80.0 * np.random.random_sample(num_refined_lasers) + 5.0
        #refined_angles1 = refined_angles1.tolist()
        #refined_angles2  = 80.0 * np.random.random_sample(num_refined_lasers) + 95.0
        #refined_angles2 = refined_angles2.tolist()
        #refined_angles3  = 80.0 * np.random.random_sample(num_refined_lasers) + 185.0
        #refined_angles3 = refined_angles3.tolist()
        #refined_angles4  = 80.0 * np.random.random_sample(num_refined_lasers) + 275.0
        #refined_angles4 = refined_angles4.tolist()
        #refined_angles = refined_angles1 + refined_angles2 + refined_angles3 + refined_angles4

        # Always consider lasers in opposite directions
        #opposite_angles = [(x+180) % 360 for x in refined_angles]
        #set1 = set(refined_angles)
        #set2 = set(opposite_angles)
        #refined_angles_set = set1.union(set2)
        #refined_angles = list(refined_angles_set)

        # Inaccurate defined angles may run into error
        #refined_angles = [math.ceil(x*10)/10 for x in refined_angles]

        # NOTE: Order of angles in laser_angles should be consistent with input image of NN 
        self.laser_angles = self.principal_angles + refined_angles
        self.laser_angles.sort()
        print 'laser_angles = ', self.laser_angles
        print 'Number of lasers = ', len(self.laser_angles)


        # Each obstacle is a line segment and is described by six numbers: 
        # NOTE: The first four numbers are endpoint coordinates x1, y1, x2, y2, where either x1<x2 or y1<y2.
        # The second last number is 0 for horizontal obstacles, 1 for vertical obstacles.
        # The last number is index of obstacles
        # num_vertices = 8

        if num_vertices == 8:
            obst0 = [0.0, 0.0, 3.0, 0.0, 0, 0]
            obst1 = [3.0, 2.0, 8.0, 2.0, 0, 1]
            obst2 = [0.0, 6.0, 5.0, 6.0, 0, 2]
            obst3 = [5.0, 8.0, 8.0, 8.0, 0, 3]
            obst4 = [0.0, 0.0, 0.0, 6.0, 1, 4]
            obst5 = [3.0, 0.0, 3.0, 2.0, 1, 5]
            obst6 = [5.0, 6.0, 5.0, 8.0, 1, 6]
            obst7 = [8.0, 2.0, 8.0, 8.0, 1, 7]
            self.obstacles = [obst0, obst1, obst2, obst3, obst4, obst5, obst6, obst7]

            # Each vertex is described by its coordinates, angle range of partition segments start from it, and the two obstacles share this vertex. 
            # NOTE: Angle bounds should be in range [0, 360) degrees.
            # If lower bound is bigger than upper bound of angle range, the angle range will be treated as 
            # two intervals, such as 180, 90 represents [180, 360) and [0, 90]
            v0 = {'x': 0.0, 'y': 0.0, 'angle_lb': 0.0, 'angle_ub': 90.0, 'obst': [obst0, obst4]}
            v1 = {'x': 3.0, 'y': 0.0, 'angle_lb': 90.0, 'angle_ub': 180.0, 'obst': [obst0, obst5]}
            v2 = {'x': 3.0, 'y': 2.0, 'angle_lb': 0.0, 'angle_ub': 270.0, 'obst': [obst1, obst5]}
            v3 = {'x': 8.0, 'y': 2.0, 'angle_lb': 90.0, 'angle_ub': 180.0, 'obst': [obst1, obst7]}
            v4 = {'x': 0.0, 'y': 6.0, 'angle_lb': 270.0, 'angle_ub': 0.0, 'obst': [obst2, obst4]}
            v5 = {'x': 5.0, 'y': 6.0, 'angle_lb': 180.0, 'angle_ub': 90.0, 'obst': [obst2, obst6]}
            v6 = {'x': 5.0, 'y': 8.0, 'angle_lb': 270.0, 'angle_ub': 0.0, 'obst': [obst3, obst6]}
            v7 = {'x': 8.0, 'y': 8.0, 'angle_lb': 180.0, 'angle_ub': 270.0, 'obst': [obst3, obst7]}
            self.vertices = [v0, v1, v2, v3, v4, v5, v6, v7]

        elif num_vertices == 10:
            obst0 = [0.0, 0.0, 3.0, 0.0, 0, 0]
            obst1 = [3.0, 0.0, 3.0, 2.0, 1, 1]
            obst2 = [3.0, 2.0, 8.0, 2.0, 0, 2]
            obst3 = [8.0, 2.0, 8.0, 3.0, 1, 3]
            obst4 = [6.0, 3.0, 8.0, 3.0, 0, 4]
            obst5 = [6.0, 3.0, 6.0, 8.0, 1, 5]
            obst6 = [5.0, 8.0, 6.0, 8.0, 0, 6]
            obst7 = [5.0, 6.0, 5.0, 8.0, 1, 7]
            obst8 = [0.0, 6.0, 5.0, 6.0, 0, 8]
            obst9 = [0.0, 0.0, 0.0, 6.0, 1, 9]
            self.obstacles = [obst0, obst1, obst2, obst3, obst4, obst5, obst6, obst7, obst8, obst9]

            v0 = {'x': 0.0, 'y': 0.0, 'angle_lb': 0.0, 'angle_ub': 90.0, 'obst': [obst0, obst9]}
            v1 = {'x': 3.0, 'y': 0.0, 'angle_lb': 90.0, 'angle_ub': 180.0, 'obst': [obst0, obst1]}
            v2 = {'x': 3.0, 'y': 2.0, 'angle_lb': 0.0, 'angle_ub': 270.0, 'obst': [obst1, obst2]}
            v3 = {'x': 8.0, 'y': 2.0, 'angle_lb': 90.0, 'angle_ub': 180.0, 'obst': [obst2, obst3]}
            v4 = {'x': 0.0, 'y': 6.0, 'angle_lb': 270.0, 'angle_ub': 0.0, 'obst': [obst8, obst9]}
            v5 = {'x': 5.0, 'y': 6.0, 'angle_lb': 180.0, 'angle_ub': 90.0, 'obst': [obst7, obst8]}
            v6 = {'x': 5.0, 'y': 8.0, 'angle_lb': 270.0, 'angle_ub': 0.0, 'obst': [obst6, obst7]}
            v7 = {'x': 6.0, 'y': 8.0, 'angle_lb': 180.0, 'angle_ub': 270.0, 'obst': [obst5, obst6]}
            v8 = {'x': 6.0, 'y': 3.0, 'angle_lb': 90.0, 'angle_ub': 0.0, 'obst': [obst4, obst5]}
            v9 = {'x': 8.0, 'y': 3.0, 'angle_lb': 180.0, 'angle_ub': 270.0, 'obst': [obst3, obst4]}
            self.vertices = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9]

        elif num_vertices == 12:
            obst0 = [2.0, 0.0, 3.0, 0.0, 0, 0]
            obst1 = [3.0, 0.0, 3.0, 2.0, 1, 1]
            obst2 = [3.0, 2.0, 8.0, 2.0, 0, 2]
            obst3 = [8.0, 2.0, 8.0, 3.0, 1, 3]
            obst4 = [6.0, 3.0, 8.0, 3.0, 0, 4]
            obst5 = [6.0, 3.0, 6.0, 8.0, 1, 5]
            obst6 = [5.0, 8.0, 6.0, 8.0, 0, 6]
            obst7 = [5.0, 6.0, 5.0, 8.0, 1, 7]
            obst8 = [0.0, 6.0, 5.0, 6.0, 0, 8]
            obst9 = [0.0, 5.0, 0.0, 6.0, 1, 9]
            obst10 = [0.0, 5.0, 2.0, 5.0, 0, 10]
            obst11 = [2.0, 0.0, 2.0, 5.0, 1, 11]
            self.obstacles = [obst0, obst1, obst2, obst3, obst4, obst5, obst6, obst7, obst8, obst9, obst10, obst11]

            v0 = {'x': 2.0, 'y': 0.0, 'angle_lb': 0.0, 'angle_ub': 90.0, 'obst': [obst0, obst11]}
            v1 = {'x': 3.0, 'y': 0.0, 'angle_lb': 90.0, 'angle_ub': 180.0, 'obst': [obst0, obst1]}
            v2 = {'x': 3.0, 'y': 2.0, 'angle_lb': 0.0, 'angle_ub': 270.0, 'obst': [obst1, obst2]}
            v3 = {'x': 8.0, 'y': 2.0, 'angle_lb': 90.0, 'angle_ub': 180.0, 'obst': [obst2, obst3]}
            v4 = {'x': 0.0, 'y': 6.0, 'angle_lb': 270.0, 'angle_ub': 0.0, 'obst': [obst8, obst9]}
            v5 = {'x': 5.0, 'y': 6.0, 'angle_lb': 180.0, 'angle_ub': 90.0, 'obst': [obst7, obst8]}
            v6 = {'x': 5.0, 'y': 8.0, 'angle_lb': 270.0, 'angle_ub': 0.0, 'obst': [obst6, obst7]}
            v7 = {'x': 6.0, 'y': 8.0, 'angle_lb': 180.0, 'angle_ub': 270.0, 'obst': [obst5, obst6]}
            v8 = {'x': 6.0, 'y': 3.0, 'angle_lb': 90.0, 'angle_ub': 0.0, 'obst': [obst4, obst5]}
            v9 = {'x': 8.0, 'y': 3.0, 'angle_lb': 180.0, 'angle_ub': 270.0, 'obst': [obst3, obst4]}
            v10 = {'x': 0.0, 'y': 5.0, 'angle_lb': 0.0, 'angle_ub': 90.0, 'obst': [obst9, obst10]}
            v11 = {'x': 2.0, 'y': 5.0, 'angle_lb': 270.0, 'angle_ub': 180.0, 'obst': [obst10, obst11]}
            self.vertices = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]


        # Obstacle and boundary
        # TODO: Need to update for more vertice cases
        self.abst_reg_obstacles = []
        A = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
        infinity = 1000
        
        x_min, x_max, y_min, y_max = 0.0, 5.0, 6.0, 8.0
        b = [-1 * x_min, x_max, -1 * y_min, y_max]
        self.abst_reg_obstacles.append({'A': A, 'b': b})

        x_min, x_max, y_min, y_max = 3.0, 8.0, 0.0, 2.0
        b = [-1 * x_min, x_max, -1 * y_min, y_max]
        self.abst_reg_obstacles.append({'A': A, 'b': b})

        # Left boundary
        x_min, x_max, y_min, y_max = -1 * infinity, 0.0, -1 * infinity, infinity
        b = [-1 * x_min, x_max, -1 * y_min, y_max]
        self.abst_reg_obstacles.append({'A': A, 'b': b})

        # Right boundary
        x_min, x_max, y_min, y_max = 8.0, infinity, -1 * infinity, infinity
        b = [-1 * x_min, x_max, -1 * y_min, y_max]
        self.abst_reg_obstacles.append({'A': A, 'b': b})

        # Top boundary
        x_min, x_max, y_min, y_max = 0.0, 8.0, 8.0, infinity
        b = [-1 * x_min, x_max, -1 * y_min, y_max]
        self.abst_reg_obstacles.append({'A': A, 'b': b})

        # Bottom boundary
        x_min, x_max, y_min, y_max = 0.0, 8.0, -1 * infinity, 0.0
        b = [-1 * x_min, x_max, -1 * y_min, y_max]
        self.abst_reg_obstacles.append({'A': A, 'b': b})
        

    def prepare_workspace(self):    
        """
        Prepare workspace for sweep algorithm 
        """
        events, segments = [], []
        event_queue  = EventQueue()

        # Create events and segments correspond to obstacle boundaries
        for obst in self.obstacles:
            if obst[4]: # vertical 
                angle = 90.0
                # Upper endpoint has bigger y-coordinate
                upper_X, upper_Y, lower_X, lower_Y = obst[2], obst[3], obst[0], obst[1]
            else:
                angle = 180.0
                # Upper endpoint has smaller x-coordinate
                upper_X, upper_Y, lower_X, lower_Y = obst[0], obst[1], obst[2], obst[3]
            new_segment = Segment(upper_X, upper_Y, lower_X, lower_Y, angle, is_principal=True, is_boundary=True)
            segments.append(new_segment)

            # Create event for upper endpoint
            upper_event = Event(upper_X, upper_Y, set(), set(), set(), is_principal=True)
            new_upper_event = event_queue.search(event_queue.root, upper_event)
            if new_upper_event is None:
                new_upper_event = upper_event
                event_queue.insert(new_upper_event)
                events.append(new_upper_event)
            new_upper_event.U.add(new_segment)

            # Create event for lower endpoint
            lower_event = Event(lower_X, lower_Y, set(), set(), set(), is_principal=True)
            new_lower_event = event_queue.search(event_queue.root, lower_event)
            if new_lower_event is None:
                new_lower_event = lower_event
                event_queue.insert(new_lower_event)
                events.append(new_lower_event)
            new_lower_event.L.add(new_segment)


        # Partition workspace by segments start from vertices, along laser directions, and in the angle range of corresponding vertices
        for vertex_index, vertex in enumerate(self.vertices):
            if vertex['angle_lb'] <= vertex['angle_ub']:
                vertex_laser_angles = [x for x in self.laser_angles if vertex['angle_lb'] <= x <= vertex['angle_ub']]
            else:
                vertex_laser_angles = [x for x in self.laser_angles if 0.0 <= x <= vertex['angle_ub']]
                vertex_laser_angles.extend([x for x in self.laser_angles if vertex['angle_lb'] <= x < 360.0])

            # Initiate endpoints of partition segments by laser max range
            # TODO: Use radians for angles instead of conversion angles repeatedly for all vertices
            laser_ends_X = [vertex['x'] + self.max_range * math.cos(math.radians(angle)) for angle in vertex_laser_angles]
            laser_ends_Y = [vertex['y'] + self.max_range * math.sin(math.radians(angle)) for angle in vertex_laser_angles]

            # Compute intersection between obstacles and partition segments start from this vertex.
            # TODO: Computation could be more efficient by considering obstacles are either vertical or horizontal.
            # TODO: Add a check to prevent partition segments overlap obstacle boundaries.
            for i in xrange(len(vertex_laser_angles)):
                for obst in self.obstacles:
                    if obst not in vertex['obst']: 
                        obst_start_X, obst_start_Y, obst_end_X, obst_end_Y = obst[0], obst[1], obst[2], obst[3]
                        end_X, end_Y = laser_ends_X[i], laser_ends_Y[i]            
                        intersection = two_segments_intersection((obst_start_X, obst_start_Y, obst_end_X, obst_end_Y), (vertex['x'], vertex['y'], end_X, end_Y))

                        if intersection:
                            intersection_X, intersection_Y = intersection[0], intersection[1]
                            laser_ends_X[i] = intersection_X
                            laser_ends_Y[i] = intersection_Y 


            # Search event correspond to this vertex
            vertex_event     = Event(vertex['x'], vertex['y'], set(), set(), set(), is_principal=True)
            new_vertex_event = event_queue.search(event_queue.root, vertex_event)
            assert new_vertex_event is not None, 'all vertex events should have been enqueued when process obstacle boundaries'
            #if new_vertex_event is None:
            #    new_vertex_event = vertex_event
            #    event_queue.insert(new_vertex_event)
            #    events.append(new_vertex_event)

            for i, angle in enumerate(vertex_laser_angles):
                is_principal = True if angle in self.principal_angles else False
                # Segement with both endpoints are vertices should only be count once in order to avoid overlapping segments
                endpoint_is_processed_vertex = False
                for j in xrange(vertex_index):
                    if is_close(laser_ends_X[i], self.vertices[j]['x']) and is_close(laser_ends_Y[i], self.vertices[j]['y']):
                        endpoint_is_processed_vertex = True
                        break

                if not endpoint_is_processed_vertex:
                    # Create an event for partition segment endpoint.
                    # NOTE: is_principal of endpoint could true for a non-principle laser if the endpoint is a vertex. 
                    # However, all vertices should have ALREADY been enqueued
                    endpoint_event     = Event(laser_ends_X[i], laser_ends_Y[i], set(), set(), set(), is_principal=is_principal)
                    new_endpoint_event = event_queue.search(event_queue.root, endpoint_event)
                    if new_endpoint_event is None:
                        new_endpoint_event = endpoint_event
                        event_queue.insert(new_endpoint_event)
                        events.append(new_endpoint_event)
                        
                    # Create a segment and update events correspond to its endpoints     
                    if 0.0 < angle <= 180.0: # vertex is lower endpoint of the new segment
                        new_segment = Segment(vertex['x'], vertex['y'], laser_ends_X[i], laser_ends_Y[i], angle, is_principal=is_principal, is_boundary=False)
                        new_vertex_event.L.add(new_segment)
                        new_endpoint_event.U.add(new_segment)
                    else: # vertex is upper endpoint of the new segment
                        aligned_angle = (angle + 180.0) % 360
                        new_segment = Segment(vertex['x'], vertex['y'], laser_ends_X[i], laser_ends_Y[i], aligned_angle, is_principal=is_principal, is_boundary=False)
                        new_vertex_event.U.add(new_segment)
                        new_endpoint_event.L.add(new_segment)
                    segments.append(new_segment)

        return events, segments, event_queue



    def find_lidar_configuration(self, subdivisions):
        """
        LiDAR configuration for each subdivision is represented by a list of obstacle indices that lasers intersect.
        NOTE: The order of intersecting obstacle indices should be same as that of lasers in laser_angles.
        - subdivisions: a list of polyhedra, in V-rep
        """    
        lidar_configs = []
        # NOTE: lidar_configs has the same order as regions in subdivisions
        for poly in subdivisions:
            # Take an interior point of the subdivision and compute intersecting obstacle indices
            interior_X = (poly[0][0] + poly[1][0] + poly[2][0]) / 3.0
            interior_Y = (poly[0][1] + poly[1][1] + poly[2][1]) / 3.0

            # TODO: Convert angle to radians out of loop
            laser_ends_X = [interior_X + self.max_range * math.cos(math.radians(angle)) for angle in self.laser_angles]
            laser_ends_Y = [interior_Y + self.max_range * math.sin(math.radians(angle)) for angle in self.laser_angles]

            configuration = []
            for i in xrange(len(self.laser_angles)):
                closest_obst_index = -1
                for obst in self.obstacles:
                    obst_start_X, obst_start_Y, obst_end_X, obst_end_Y = obst[0], obst[1], obst[2], obst[3]
                    end_X, end_Y = laser_ends_X[i], laser_ends_Y[i]            
                    intersection = two_segments_intersection((obst_start_X, obst_start_Y, obst_end_X, obst_end_Y), (interior_X, interior_Y, end_X, end_Y))

                    if intersection:
                        intersection_X, intersection_Y = intersection[0], intersection[1]
                        laser_ends_X[i] = intersection_X
                        laser_ends_Y[i] = intersection_Y 
                        closest_obst_index = obst[5]

                assert closest_obst_index != -1, 'No intersecting obstacle is found when compute LiDAR configuration'
                configuration.append(closest_obst_index)          

            lidar_configs.append(configuration)    

        return lidar_configs

