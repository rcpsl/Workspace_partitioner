# Find intersection between s (endpoints a,b) and t (endpoints c,d)
aX, aY = 0.0, 6.0
bX, bY = 4.0, 2.0
cX, cY = 2.5, 6.0
dX, dY = 3.0, 2.0


A = (bX - aX) * (cY - dY) - (cX - dX) * (bY - aY)

if A != 0.0:
    As = (cX - aX) * (cY - dY) - (cX - dX) * (cY - aY)
    At = (bX - aX) * (cY - aY) - (cX - aX) * (bY - aY)
    ks = As / A
    kt = At / A

intersection_X = (1 - ks) * aX + ks * bX
intersection_Y = (1 - ks) * aY + ks * bY  
print 'intersection', (intersection_X, intersection_Y)  