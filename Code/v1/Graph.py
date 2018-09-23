class Vertex(object):
    def __init__(self, event_coordinates):
        self.id = event_coordinates
        self.adjacent = {}

    def __str__(self):
         return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, attribute):
        self.adjacent[neighbor] = attribute

    def get_connections(self):
        return self.adjacent.keys()



class Graph(object):
    def __init__(self):
        self.vertex_dict = {}
        self.num_vertices = 0
        self.num_edges    = 0

    def __iter__(self):
        return iter(self.vertex_dict.values())

    def add_vertex(self, event_coordinates):
        self.num_vertices += 1
        new_vertex = Vertex(event_coordinates)
        self.vertex_dict[event_coordinates] = new_vertex

    def add_edge(self, frm, to, attribute):
        """
        attribute is a dictionary stores following attributes associated with each edge:
        - orientation: angle of directed edge relative to x-positive, in range (0, 360] degrees
        - is_boundary:  boolean as defined in Segment class
        """
        self.num_edges += 1
        self.vertex_dict[frm].add_neighbor(self.vertex_dict[to], attribute)