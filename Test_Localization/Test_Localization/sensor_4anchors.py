import random
from array import *
import numpy as np

#                       x        y       z        dt
anchor_A = array('f', [0       ,0       ,0       ,0.5])
anchor_B = array('f', [100     ,500     ,-2      ,0.5])
anchor_C = array('f', [400     ,0       ,0.5     ,0.3])
anchor_D = array('f', [-800    ,50      ,-0.2    ,0.5])
sensor_S = array('f', [500     ,500    ,-500])
lists = [anchor_A, anchor_B, anchor_C, anchor_D]
velocity = 1500
depth = sensor_S[2]

def getDistance(a, b):
    distance = ((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)**0.5
    return distance

def dtCalc(a, b, c, d):
    dt = array('f', [])
    #dtb
    time = b[3]
    dt.append(time)
    #dtc
    time = (getDistance(a,b)+getDistance(b,c)-getDistance(a,c))/velocity + c[3] + b[3]
    time += time*np.random.normal(0,1e-3)
    dt.append(time)
    #dtd
    time = (getDistance(a,b)+getDistance(b,c)+getDistance(c,d)-getDistance(a,d))/velocity + d[3] + c[3] + b[3]
    time += time*np.random.normal(0,1e-3)
    dt.append(time)
    #dt1
    time = (getDistance(a,b)+getDistance(b,sensor_S)-getDistance(a,sensor_S))/velocity + b[3]
    time += time*np.random.normal(0,1e-3)
    dt.append(time)
    #dt2
    time = (getDistance(a,b)+getDistance(b,c)+getDistance(c,sensor_S)-getDistance(a,sensor_S))/velocity + c[3] + b[3]
    time += time*np.random.normal(0,1e-3)
    dt.append(time)
    #dt3
    time = (getDistance(a,b)+getDistance(b,c)+getDistance(c,d)+getDistance(d,sensor_S)-getDistance(a,sensor_S))/velocity + d[3] + c[3] + b[3]
    time += time*np.random.normal(0,1e-3)
    dt.append(time)
    #dt = [dtb, dtc, dtd, dt1, dt2, dt3]
    #       0    1    2    3    4    5
    return dt

def calc(ans):
    dt = array('f', [])
    dt = dtCalc(ans[0], ans[1], ans[2], ans[3])
    dt1_b = dt[3] - dt[0]
    dt2_c = dt[4] - dt[1]
    dt3_d = dt[5] - dt[2]
    ab = getDistance(ans[0], ans[1])
    ac = getDistance(ans[0], ans[2])
    ad = getDistance(ans[0], ans[3])
    k1 = ab - dt1_b*velocity
    k2 = ac - dt2_c*velocity
    k3 = ad - dt3_d*velocity
    M11 = 2*(ans[0][0] - ans[1][0])
    M12 = 2*(ans[0][1] - ans[1][1])
    M13 = 2*(ans[0][2] - ans[1][2])
    M21 = 2*(ans[0][0] - ans[2][0])
    M22 = 2*(ans[0][1] - ans[2][1])
    M23 = 2*(ans[0][2] - ans[2][2])
    M31 = 2*(ans[0][0] - ans[3][0])
    M32 = 2*(ans[0][1] - ans[3][1])
    M33 = 2*(ans[0][2] - ans[3][2])
    C1 = -2*k1
    C2 = -2*k2
    C3 = -2*k3
    D1 = -(ans[1][0]**2 + ans[1][1]**2 + ans[1][2]**2 - k1**2 - ans[0][0]**2 - ans[0][1]**2 - ans[0][2]**2) 
    D2 = -(ans[2][0]**2 + ans[2][1]**2 + ans[2][2]**2 - k2**2 - ans[0][0]**2 - ans[0][1]**2 - ans[0][2]**2) 
    D3 = -(ans[3][0]**2 + ans[3][1]**2 + ans[3][2]**2 - k3**2 - ans[0][0]**2 - ans[0][1]**2 - ans[0][2]**2)
    N11 = M22*M33-M32*M23
    N12 = M32*M13-M12*M31
    N13 = M12*M23-M22*M13
    N21 = M31*M23-M33*M21
    N22 = M11*M33-M31*M13
    N23 = M21*M13-M11*M23
    N31 = M21*M32-M31*M22
    N32 = M12*M31-M11*M32
    N33 = M22*M11-M12*M21
    N = M11*M22*M33 + M12*M23*M31 + M13*M21*M32 - M11*M23*M32 - M12*M21*M33 - M13*M22*M31
    if N==0:
        print('no proper answer')
    else:
        N1 = C1*M22*M33 + M12*M23*C3 + M13*C2*M32 - C1*M23*M32 - M12*C2*M33 - M13*M22*C3
        N2 = M11*C2*M33 + C1*M23*M31 + M13*M21*C3 - M11*M23*C3 - C1*M21*M33 - M13*C2*M31
        N3 = M11*M22*C3 + M12*C2*M31 + C1*M21*M32 - M11*C2*M32 - M12*M21*C3 - C1*M22*M31
        N4 = D1*M22*M33 + M12*M23*D3 + M13*D2*M32 - D1*M23*M32 - M12*D2*M33 - M13*M22*D3
        N5 = M11*D2*M33 + D1*M23*M31 + M13*M21*D3 - M11*M23*D3 - D1*M21*M33 - M13*D2*M31
        N6 = M11*M22*D3 + M12*D2*M31 + D1*M21*M32 - M11*D2*M32 - M12*M21*D3 - D1*M22*M31
        A1 = N1/N
        A2 = N2/N
        A3 = N3/N
        B1 = N4/N
        B2 = N5/N
        B3 = N6/N
        alpha = A1**2 + A2**2 + A3**2 - 1
        beta = 2*(A1*(B1-ans[0][0]) + A2*(B2-ans[0][1]) + A3*(B3-ans[0][2]))
        gamma = B1**2 + B2**2 + B3**2 + ans[0][0]**2 + ans[0][1]**2 + ans[0][2]**2
        sa1 = (-beta-(beta**2-4*alpha*gamma)**0.5)/(2*alpha)
        sa2 = (-beta+(beta**2-4*alpha*gamma)**0.5)/(2*alpha)
        if not isinstance(sa1,complex):
            if sa1 > 0:
                sa = sa1
            else:
                sa = sa2
            xs = A1*sa + B1
            ys = A2*sa + B2
            zs = -(A3*sa + B3)
            xse = abs((xs-sensor_S[0])/sensor_S[0])*100
            yse = abs((ys-sensor_S[1])/sensor_S[1])*100
            zse = abs((zs-sensor_S[2])/sensor_S[2])*100

            print(str(xs),' ',str(ys),' ',str(zs))
            print('error in x: ',str(xse),'%')
            print('error in y: ',str(yse),'%')
            print('error in z: ',str(zse),'%')
        else:
            print('no proper answer')

calc(lists)