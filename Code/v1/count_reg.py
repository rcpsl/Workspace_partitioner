import pickle
from Workspace import *

"""
filename = './results/refined_reg_H_rep_dict.txt'
with open(filename, 'rb') as inputFile:
    refined_reg_H_rep_dict = pickle.load(inputFile)

total_refined_regions = 0
for i, refined_reg_H_rep_in_this_abst in refined_reg_H_rep_dict.items():
    num_refined_reg = len(refined_reg_H_rep_in_this_abst)
    total_refined_regions += num_refined_reg
    print 'abstract regtion', i, '--> ', num_refined_reg, 'refined regions'

print 'Total refined regions = ', total_refined_regions      
"""

filename = './results/region10-15/abst_reg_H_rep.txt'
with open(filename, 'rb') as inputFile:
    abs_reg_H_rep = pickle.load(inputFile)
#print abs_reg_H_rep

workspace = Workspace()
#print workspace.abst_reg_obstacles

abs_reg_H_rep_with_obstacles = abs_reg_H_rep + workspace.abst_reg_obstacles
print '\n'
print len(abs_reg_H_rep_with_obstacles)


outputFileName = 'results/abst_reg_H_rep_with_obstacles.txt'
with open(outputFileName, 'wb') as outputFile:
    pickle.dump(abs_reg_H_rep_with_obstacles, outputFile)
outputFile.close()

"""
filename = './results/regions10-13/abst_reg_V_rep.txt'
with open(filename, 'rb') as inputFile:
    abst_reg_V_rep = pickle.load(inputFile)
    for abst_reg in abst_reg_V_rep:
        print abst_reg
"""        