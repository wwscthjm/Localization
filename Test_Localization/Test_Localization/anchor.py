import random
from array import *

#                       x        y       z        dt
anchor_A = array('f', [0       ,0       ,0       ,0.5])
anchor_B = array('f', [100     ,500     ,-2      ,0.5])
anchor_C = array('f', [400     ,0       ,0.5     ,0.3])
anchor_D = array('f', [-800    ,50      ,-0.2    ,0.5])
anchor_E = array('f', [500     ,200     ,0.6     ,0.4])
anchor_F = array('f', [0       ,500     ,-0.9    ,0.5])
anchor_G = array('f', [0       ,600     ,0       ,0.9])
anchor_H = array('f', [-400    ,100     ,0       ,0.6])
anchor_I = array('f', [300     ,500     ,0.7     ,0.5])
anchor_J = array('f', [-200    ,900     ,1       ,0.5])
sensor_S = array('f', [500     ,-100    ,-2000])
lists = [anchor_A,anchor_B,anchor_C,anchor_D,anchor_E,anchor_F,anchor_G,anchor_H]
velocity = 1500
depth = sensor_S[2]

def getSubLists(lis=[], m=0):
    allAns = []
    ans = [None for i in range(m)]    
    subLists(lis,m,ans,allAns)
    return allAns

def subLists(lis=[], m=0, ans=[], allAns=[]):
    # recursive function  codes
    if m == 0:
        allAns.append(ans.copy()) 
        return
    length = len(lis)
    for iter in range(length-m+1):
        ans[-m] = lis[iter]
        if iter + 1 < length:
            subLists(lis[iter+1:], m-1, ans, allAns)
        else:
            allAns.append(ans.copy())
            return

def getDistance(a, b):
    distance = ((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)**0.5
    return distance

def dtCalc(a, b, c):
    dt = array('f', [])
    #dtb
    time = b[3]
    dt.append(time)
    #dtc
    time = (getDistance(a,b)+getDistance(b,c)-getDistance(a,c))/velocity + c[3] + b[3]
    dt.append(time)
    #dt1
    time = (getDistance(a,b)+getDistance(b,sensor_S)-getDistance(a,sensor_S))/velocity + b[3]
    dt.append(time)
    #dt2
    time = (getDistance(a,b)+getDistance(b,c)+getDistance(c,sensor_S)-getDistance(a,sensor_S))/velocity + c[3] + b[3]
    dt.append(time)
    #dt = [dtb, dtc, dt1, dt2]
    return dt

def calc(ans):
    dt = array('f', [])

    dt = dtCalc(ans[0], ans[1], ans[2])
    dt1_b = dt[2] - dt[0]
    dt2_c = dt[3] - dt[1]
    ab = getDistance(ans[0], ans[1])
    ac = getDistance(ans[0], ans[2])
    k1 = dt1_b*velocity - ab
    k2 = dt2_c*velocity - ac
        
    Z1 = (depth - ans[0][2])**2 - (depth - ans[1][2])**2
    Z2 = (depth - ans[0][2])**2 - (depth - ans[2][2])**2
    A = k1**2 + ans[0][0]**2 - ans[1][0]**2 + ans[0][1]**2 - ans[1][1]**2 + Z1
    B = k2**2 + ans[0][0]**2 - ans[2][0]**2 + ans[0][1]**2 - ans[2][1]**2 + Z2
    M = 2*((ans[0][1]-ans[1][1])*(ans[0][0]-ans[2][0]) - (ans[0][1]-ans[2][1])*(ans[0][0]-ans[1][0]))
    if M==0:
        print('no proper answer')
    else:
        Mx = 2*((ans[0][1]-ans[1][1])*k2 - (ans[0][1]-ans[2][1])*k1)/M
        Nx = ((ans[0][1]-ans[1][1])*B - (ans[0][1]-ans[2][1])*A)/M
        My = 2*(k1*(ans[0][0]-ans[2][0]) - k2*(ans[0][0]-ans[1][0]))/M
        Ny = (A*(ans[0][0]-ans[2][0]) - B*(ans[0][0]-ans[1][0]))/M
        Nxp = Nx - ans[0][0]
        Nyp = Ny - ans[0][1]
        alpha = Mx**2 + My**2 - 1
        beta = 2*(Mx*Nxp + My*Nyp)
        gamma = Nxp**2 + Nyp**2 + (sensor_S[2]-ans[0][2])**2
        sa1 = (-beta-(beta**2-4*alpha*gamma)**0.5)/(2*alpha)
        sa2 = (-beta+(beta**2-4*alpha*gamma)**0.5)/(2*alpha)
        if not isinstance(sa1,complex):
            if sa1 > 0:
                sa = sa1
            else:
                sa = sa2
            xs = Mx*sa + Nx
            ys = My*sa + Ny
            coordinate = array('f', [xs, ys, depth])
            #print('(' + str(xs) + ', ' +  str(ys) + ', ' +  str(depth) + ')')
            print(coordinate.tolist())
        else:
            print('no proper answer')

ans = getSubLists(lists,3)
for i in range(len(ans)):
    print(str(i),':')
    calc(ans[i])