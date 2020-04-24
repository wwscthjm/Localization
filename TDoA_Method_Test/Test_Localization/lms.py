# computes location of sensor [estimation] ~ distance between sensor and anchor nodes
# localization algorithm : solve linear system. exact distance without error. (x)
# localization algorithm : linear least square.  (error considered) multilateration
# minimize the sum of (distance between anchors and estimated sensor node position) residual's mean/median
# [-]solution: gauss newton iteration . initial from linear least square.
# [-]remove outlier: least median squares


import numpy as np
from scipy.optimize import least_squares
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

# Test point is 2000, 2000, -100
s1=(2000, 2000, -100)
#anchor nodes position (xn, yn, zn)
p1 = (1000.0, 1000.0, -800)
p2 = (1000.0, 2000.0, -700)
p3 = (1000.0, 3000.0, -600)
p4 = (2000.0, 1000.0, -350)
p5 = (2000.0, 2000.0, -800)
p6 = (2000.0, 3000.0, -350)
p7 = (3000.0, 1000.0, -600)
p8 = (3000.0, 2000.0, -700)
p9 = (3000.0, 3000.0, -500)

#real distance between 9 anchor and TEST sensor(pre-calculated)
d =(1577.97338380595,1166.19037896906,1500.0,1030.7764064044152,700.0,1030.7764064044152,1500.0,1166.19037896906,1469.6938456699068)
#   *(1+0.05*np.random.normal(0,1))

#compute real distance
# def realdistance(xi,yi,zi,xs,ys,zs):
#     dreal = np.sqrt((xi-xs)**2+(yi-ys)**2+(zi-zs)**2) # matrix subtraction?
#     return dreal

# only add error to one distance  +np.random.normal(0,1)
def equations(guess):
    x, y, z, r = guess

    return (
        (x - p1[0]) ** 2 + (y - p1[1]) ** 2 + (z - p1[2]) ** 2 - (d[0] - r) ** 2,
        (x - p2[0]) ** 2 + (y - p2[1]) ** 2 + (z - p2[2]) ** 2 - (d[1] - r) ** 2,
        (x - p3[0]) ** 2 + (y - p3[1]) ** 2 + (z - p3[2]) ** 2 - (d[2] - r) ** 2,
        (x - p4[0]) ** 2 + (y - p4[1]) ** 2 + (z - p4[2]) ** 2 - (d[3] - r) ** 2,
        (x - p5[0]) ** 2 + (y - p5[1]) ** 2 + (z - p5[2]) ** 2 - (d[4] - r) ** 2,
        (x - p6[0]) ** 2 + (y - p6[1]) ** 2 + (z - p6[2]) ** 2 - (d[5] - r) ** 2,
        (x - p7[0]) ** 2 + (y - p7[1]) ** 2 + (z - p7[2]) ** 2 - (d[6] - r) ** 2,
        (x - p8[0]) ** 2 + (y - p8[1]) ** 2 + (z - p8[2]) ** 2 - (d[7] - r) ** 2,
        (x - p9[0]) ** 2 + (y - p9[1]) ** 2 + (z - p9[2]) ** 2 - (d[8] - r) ** 2,
    )

##################this is a print test part######################
#dreal = realdistance(x9[0],x9[1],x9[2],s[0],s[1],s[2])
#print(dreal)
################################################################

initial_guess = (0,0,0,0)
results = least_squares(equations, initial_guess)
print('least square',results.x)


####################################plot###################################
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#real sensor position
# x=[s1[0]]
# y=[s1[1]]
# z=[s1[2]]
# ax.scatter(x, y, z, c='g', marker='o')
#sensor estimation position
# x=[results.x[0]]
# y=[results.x[1]]
# z=[results.x[2]]
# ax.scatter(x, y, z, c='r', marker='1')
# #trajectory
# x=[p1[0],p2[0],p3[0],p4[0],p5[0],p9[0]]
# y=[p1[1],p2[1],p3[1],p4[1],p5[1],p9[1]]
# z=[p1[2],p2[2],p3[2],p4[2],p5[2],p9[2]]
# ax.scatter(x, y, z, c='b', marker='<')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()
