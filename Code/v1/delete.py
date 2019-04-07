import pickle
import numpy as np
from Workspace import Workspace
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import math
from keras.models import load_model
#from shapely.geometry import Point
#import shapely.geometry.polygon  as shapely


#(1,0.5)(5,5.5)
def add_lidar_image_constraints():          
        """
        For a certain laser i, if it intersects a vertical obstacle:
            x_i = x_obstacle
            y_i = y_car + (x_obstacle - x_car) tan(laser_angle)
        Otherwise:
            x_i = x_car + (y_obstacle - y_car) cot(laser_angle)
            y_i = y_obstacle
        """

        #print lidar_config
        image = [7.0, 0.6879858763864091, -0.68798587638641, -1.0, -0.5000000000000002, -0.06254417058058293, 0.06254417058058293, 0.4999999999999998, 6.999999999999998, 5.5, 5.5, 1.0000000000000002, -0.5, -0.5, -0.5, -0.5]
        image = [1.5, 0.375265023483496, -0.37526502348349633, -1.0, -0.5000000000000002, -0.06254417058058293, 0.06254417058058293, 0.4999999999999998, 1.4999999999999993, 3.0, 3.0, 1.0000000000000002, -0.5, -0.5, -0.5, -0.5]
        reg = in_region(abst_reg_H_rep,np.array([1,0.5]))
        print(reg)
        lidar_config_r = lidar_config[reg]
        for i in xrange(8):
            # NOTE: Difference between indices of x,y coordinates for the same laser in image is number of lasers
            # rVars = [solver.rVars[varMap['image'][i]], solver.rVars[varMap['image'][i+self.num_lasers]], 
            #         solver.rVars[varMap['current_state']['x']], solver.rVars[varMap['current_state']['y']]]
            rVars = [image[i], image[i+8],1, 0.5]        
            placement = obstacles[lidar_config_r[i]][4]
            angle     = workspace.laser_angles[i]            
            # TODO: tan, cot do work for horizontal and vertical lasers.
            # TODO: Convert angles to radians out of loop.
            # TODO: Better way to compute cot, maybe numpy.
            if placement: # obstacle is vertical
                obst_x    = obstacles[lidar_config_r[i]][0]
                tan_angle = math.tan(math.radians(angle))
                A = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, tan_angle, 0.0]]
                #A = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, tan_angle, -1.0]]
                b = [obst_x, obst_x * tan_angle]

            else: # obstacle is horizontal
                obst_y    = obstacles[lidar_config_r[i]][1]
                cot_angle = math.cos(math.radians(angle)) / math.sin(math.radians(angle))
                A = [[1.0, 0.0, 0.0, cot_angle], [0.0, 1.0, 0.0, 1.0]]     
                #A = [[1.0, 0.0, -1.0, cot_angle], [0.0, 1.0, 0.0, 0.0]]     
                b = [obst_y * cot_angle, obst_y]

            ret = (np.array(A).dot(np.array(rVars)))
            for (v1,v2) in  zip(ret,b):
                print(v1,v2)

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


filename = './regions/regions10-15/abst_reg_H_rep.txt'
with open(filename, 'rb') as inputFile:
    abst_reg_H_rep = pickle.load(inputFile)

filename = './regions/regions10-15/abst_reg_V_rep.txt'
with open(filename, 'rb') as inputFile:
    abst_reg_V_rep = pickle.load(inputFile)

filename = './regions/regions10-15/lidar_config_dict.txt'
with open(filename, 'rb') as inputFile:
    lidar_config = pickle.load(inputFile)
workspace = Workspace()
obstacles = workspace.obstacles
config = workspace.find_lidar_configuration(abst_reg_V_rep)
# print(config[18])
# print(in_region(abst_reg_H_rep, np.array([1,0.5])))
# print(in_region(abst_reg_H_rep, np.array([5.5,5.4])))
# print(len(abst_reg_V_rep))
# print(abst_reg_V_rep[1])
#fig, ax = plt.subplots()

patches = []
centers = []
idxs = []
sh_polygons = []
#for idx,region in enumerate(abst_reg_V_rep):
#    # print(np.array(region).shape)
#    pts = np.array(region)
#    p = Polygon(pts, closed=True)
#    centers.append(np.mean(pts,axis = 0).tolist())
#    idxs.append(idx)
#    patches.append(p)
#    pts_l = list(pts)
#    sh_polygons.append(shapely.Polygon(pts_l))

#for line in obstacles:
#    l = Line2D([line[0],line[2]],[line[1],line[3]],linestyle ='-',color='r')
#    ax.add_line(l)
#
#p = PatchCollection(patches, alpha=0.4)
#ax.set_xlim([0,6])
#ax.set_ylim([0,6])
#ax.add_collection(p)
#for i,center in enumerate(centers):
#    ax.annotate(idxs[i], xy=(0,0), xytext=center)  

points = [
[1.0, 0.5],
[1.0, 0.5],
[0.9961953978054225, 0.5262260586023331],
[0.9885861934162676, 0.5786781758069992],
[0.9775123307481408, 0.6567219961434603],
[0.9637009580619633, 0.7590383868664503],
[0.9482565307989717, 0.8836337123066187],
[0.9326433925889432, 1.0278630834072828],
[0.9186457917094231, 1.1884678583592176],
[0.9061388080008328, 1.3606572076678276],
[0.8972459780052304, 1.540786549448967],
[0.8942641676403582, 1.7249570898711681],
[0.8996185166761279, 1.9090407453477383],
[0.9152800929732621, 2.0893590189516544],
[0.9428894994780421, 2.2625659219920635],
[0.9838070389814675, 2.425289113074541],
[1.03863590862602, 2.5740060806274414],
[1.1075553675182164, 2.70551872625947],
[1.1906667659059167, 2.818115968257189],
[1.288100101519376, 2.9111906439065933],
[1.3998131798580289, 2.9842629842460155],
[1.5256464802660048, 3.0370048619806767],
[1.665299599058926, 3.069250039756298],
[1.8183683888055384, 3.0809961147606373],
[1.984351516701281, 3.07238345220685],
[2.1626624041236937, 3.0437031984329224],
[2.3529163086786866, 2.995105654001236],
[2.554250242654234, 2.9287459328770638],
[2.7628478622063994, 2.8546048291027546],
[2.968459435738623, 2.791932249441743],
[3.170437623746693, 2.74093078635633],
[3.365459445863962, 2.711602319031954],
[3.552753647323698, 2.70485389046371],
[3.7222676491364837, 2.7153941728174686],
[3.87547665880993, 2.7442704141139984],
[4.0137750050053, 2.792049204930663],
[4.138500799890608, 2.8587875477969646],
[4.257579375989735, 2.946180544793606],
[4.372470545582473, 3.056202696636319],
[4.483927866443992, 3.192846853286028],
[4.593024236615747, 3.359132817015052],
[4.701325273141265, 3.551184656098485],
[4.810761738102883, 3.763171961531043],
[4.922289588488638, 3.989632476121187],
[5.036766949575394, 4.224866783246398],
[5.155111110769212, 4.463015913963318],
[5.278215062338859, 4.698247259482741],
[5.406533730216324, 4.924897935241461],
[5.5391143136657774, 5.140547255054116],
[5.6717265117913485, 5.355185568332672],
[5.804400615859777, 5.5865098256617785],
[5.939319574274123, 5.826360264793038],
[6.075958113651723, 6.0702896397560835],
[6.211337951011956, 6.325195070356131],
[6.395757979247719, 6.475859181955457],
[6.6325872875750065, 6.513581216335297],
[6.926049490924925, 6.427006995305419],
[7.281503128819168, 6.201694946736097],
[7.712974094320089, 5.800674455240369],
[8.221209889277816, 5.223020873963833]
]
# add_lidar_image_constraints()
for idx,pt in enumerate(points):
    print(idx,pt,',region:',in_region(abst_reg_H_rep,np.array(pt)))
#for idx,pt in enumerate([[2.0621908059359213, 3.5000000000000004]]):
#    for i,poly in enumerate(sh_polygons):
#        if(poly.contains(Point((pt[0],pt[1])))):
#            print(idx,i)
#            break
#    print(-1)
    
# plt.show()
# model = load_model("model/my_model.h5")
# image_in = np.array([1.5517434692010283, 0.3272769187975614, -0.3272769187975617, -0.9482565307989717, -0.8836337123066189, -0.11053227526651785, 0.11053227526651753, 0.8836337123066184, 1.551743469201028, 2.6163662876933813, 2.6163662876933813, 0.9482565307989719, -0.8836337123066187, -0.8836337123066187, -0.8836337123066187, -0.8836337123066187]).reshape((1,-1))
# print(model.predict(image_in))

#print(in_region(abst_reg_H_rep,np.array([2.0621908059359213, 3.5000000000000004]))) 

