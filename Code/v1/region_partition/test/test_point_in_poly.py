from time import time
import numpy as np
import matplotlib.path as mpltPath

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# ---------------------------- Check Correctness --------------------#
coord = (-0.99999, 0.0)

# Matplotlib
point = [coord]
#polygon = [[-1.0, -2.0], [1.0, -1.0], [1.5, 1.0], [0.5, 1.2], [-1.0, 0.0]]
polygon = [(-1.0, -2.0), (1.0, -1.0), (1.5, 1.0), (0.5, 1.2), (-1.0, 0.0)]
path = mpltPath.Path(polygon)
print 'Matplotlib result = ', path.contains_points(point)


# Shapely
point = Point(coord)
polygon = Polygon([(-1.0, -2.0), (1.0, -1.0), (1.5, 1.0), (0.5, 1.2), (-1.0, 0.0)])
print 'shapely result = ', polygon.contains(point)


# -------------------------- Compare Performance -------------------- #
N = 10000
points = zip(np.random.random(N),np.random.random(N))


# Matplotlib
lenpoly = 100
polygon = [[np.sin(x)+0.5, np.cos(x)+0.5] for x in np.linspace(0,2*np.pi,lenpoly)[:-1]]

start_time = time()
path = mpltPath.Path(polygon)
inside = path.contains_points(points)
# NOTE: use for loop for fair comparison
#for coord in points:
#    point = [coord]
#    inside = path.contains_points(point)
print "Matplotlib Elapsed time: " + str(time()-start_time)


# Shapely
lenpoly = 100
polygon = Polygon([(np.sin(x)+0.5, np.cos(x)+0.5) for x in np.linspace(0,2*np.pi,lenpoly)[:-1]])

# TODO: how to vectorize?
start_time = time()
for coord in points:
    point = Point(coord)
    inside = polygon.contains(point)
print "Shapely Elapsed time: " + str(time()-start_time)