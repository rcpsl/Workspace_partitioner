#import sys
#sys.path.append("/usr/local/lib/python2.7/site-packages")
import cdd

"""
mat = cdd.Matrix([[2,-1,-1,0],[0,1,0,0],[0,0,1,0]], number_type='fraction')
mat.rep_type = cdd.RepType.INEQUALITY
poly = cdd.Polyhedron(mat)
print poly

ext = poly.get_generators()
print ext

print(list(ext.lin_set))
"""

"""
mat = cdd.Matrix([[6, -1, 0], [-1, 1, 0], [5, 0, -1], [-2, 0, 1]], number_type='float')
mat.rep_type = cdd.RepType.INEQUALITY
poly = cdd.Polyhedron(mat)
print poly

ext = poly.get_generators()
print ext

print(list(ext.lin_set))
"""

"""
mat = cdd.Matrix([[1, 1, 5], [1, 1, 2], [1, 6, 2], [1, 6, 5]], number_type='float')
mat.rep_type = cdd.RepType.GENERATOR
poly = cdd.Polyhedron(mat)
print poly

ine = poly.get_inequalities()
print ine
"""


vertices = [[1, 6.0, 6.0], [1, 8.0, 8.0], [1, 10.0, 8.0], [1, 12.0, 6.0], [1, 10.0, 4.0], [1, 8.0, 4.0]]
mat = cdd.Matrix(vertices, number_type='float')
poly = cdd.Polyhedron(mat)

ine = poly.get_inequalities()
#ine.canonicalize()

#ine = zip(*list(ine[:]))
#b = list(ine[0])

A, b = [], []
for row in ine:
    b.append(row[0])
    a = [-x for x in list(row[1:])]
    A.append(a)
print 'b = ', b
print 'A = ', A    



# Inflate polyhedron a little bit in H-rep and transfer back to V-rep
# Ax <= b + s *1 (where 1 is a vector of all ones)
abs_tol = 0.1
ine = []
for i, param in enumerate(b):
    param += abs_tol
    row = [param] + [-x for x in A[i]]
    ine.append(row)

mat = cdd.Matrix(ine, number_type='float')
mat.rep_type = cdd.RepType.INEQUALITY
poly = cdd.Polyhedron(mat)
ext = poly.get_generators()
poly_inflated = []
for vertex in ext:
    poly_inflated.append((vertex[1], vertex[2]))
print poly_inflated

