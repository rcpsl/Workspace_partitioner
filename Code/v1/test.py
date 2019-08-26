import pickle
import numpy as np
from Workspace import Workspace
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import math
from keras.models import load_model
import cdd
from shapely.geometry import Point
import shapely.geometry.polygon  as shapely
from shapely import geometry

def in_region(regions,x):
    
    ret = -1
    for idx,region in enumerate(regions):
        H = np.array(region['A'])
        b = np.array(region['b'])
        if(np.less(H.dot(x),b).all()):
            # print(idx)
            if(ret != -1):
              return -2
            else:
                ret = idx
    return ret

def plotPolygon(polygon, text, axis):
        x, y = polygon.exterior.xy
        axis.plot(x, y)

        center = polygon.centroid.coords[:][0]  # region center
        axis.text(center[0], center[1], text)


filename = './regions/regions10-15/abst_reg_H_rep.txt'
filename = './results/abst_reg_H_rep.txt'
with open(filename, 'rb') as inputFile:
    abst_reg_H_rep = pickle.load(inputFile)

filename = './regions/regions10-15/abst_reg_V_rep.txt'
filename = './results/abst_reg_V_rep.txt'
with open(filename, 'rb') as inputFile:
    abst_reg_V_rep = pickle.load(inputFile)

filename = './regions/regions10-15/lidar_config_dict.txt'
filename = './results/lidar_config_dict.txt'

with open(filename, 'rb') as inputFile:
    lidar_config = pickle.load(inputFile)
# workspace = Workspace()
# obstacles = workspace.obstacles
# config = workspace.find_lidar_configuration(abst_reg_V_rep)
# print(config[18])
# print(in_region(abst_reg_H_rep, np.array([1,0.5])))
# print(in_region(abst_reg_H_rep, np.array([5.5,5.4])))
# print(len(abst_reg_V_rep))
# print(abst_reg_V_rep[1])
fig, ax = plt.subplots()

patches = []
centers = []
sh_polygons = []
polygons =[]
for idx,region in enumerate(abst_reg_V_rep):
   # print(np.array(region).shape)
   pts = np.array(region)
   poly = geometry.Polygon([[p[0],p[1]] for p in pts])
   polygons.append(poly)
   p_plt = Polygon(pts, closed=True)
   centers.append([poly.centroid.x,poly.centroid.y])
   patches.append(p_plt)
   pts_l = list(pts)
   sh_polygons.append(shapely.Polygon(pts_l))
   

# for line in obstacles:
#    l = Line2D([line[0],line[2]],[line[1],line[3]],linestyle ='-',color='r')
#    ax.add_line(l)

ax.set_xlim([0,6])
ax.set_ylim([0,6])
# ax.add_collection(p_plt)
for i,poly in enumerate(polygons):
#    ax.annotate(idxs[i], xy=(0,0), xytext=center)  
    plotPolygon(poly, i, ax)

plt.show()

# add_lidar_image_constraints()
# for idx,pt in enumerate(points):
#     print(idx,pt,',region:',in_region(abst_reg_H_rep,np.array(pt)))
#for idx,pt in enumerate([[2.0621908059359213, 3.5000000000000004]]):
#    for i,poly in enumerate(sh_polygons):
#        if(poly.contains(Point((pt[0],pt[1])))):
#            print(idx,i)
#            break
#    print(-1)
    
# model = load_model("model/my_model.h5")
# image_in = np.array([1.5517434692010283, 0.3272769187975614, -0.3272769187975617, -0.9482565307989717, -0.8836337123066189, -0.11053227526651785, 0.11053227526651753, 0.8836337123066184, 1.551743469201028, 2.6163662876933813, 2.6163662876933813, 0.9482565307989719, -0.8836337123066187, -0.8836337123066187, -0.8836337123066187, -0.8836337123066187]).reshape((1,-1))
# print(model.predict(image_in))

#print(in_region(abst_reg_H_rep,np.array([2.0621908059359213, 3.5000000000000004]))) 

# model = load_model("model/my_model.h5")
# # When num_layers is 4: [image_size, hidden_layer_size, hidden_layer_size, hidden_layer_size, last_layer_size]
# layer_sizes = [16] + [layer.units for layer in model.layers]
# print('hi')

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