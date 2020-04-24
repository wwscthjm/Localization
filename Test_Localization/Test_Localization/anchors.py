from files import *
import numpy as np
from scipy.optimize import least_squares

anchor_A = {'x': 0, 'y': 0, 'z': 0}				#A:(0,0,0)
anchor_B = {'x': 100, 'y': 0, 'z': 0}			#Xb>0
anchor_E = {'x': 400, 'y': 0, 'z': 0}			#Xe>0
anchor_H = {'x': 800, 'y': 0, 'z': 0}			#Xh>0
anchor_C = {'x': 0, 'y': 100, 'z': 0}			#Yc>0
anchor_F = {'x': 0, 'y': 500, 'z': 0}			#Yf>0
anchor_I = {'x': 0, 'y': 600, 'z': 0}			#Yi>0
anchor_D = {'x': 100, 'y': 100, 'z': 1000}		#Zd>0
anchor_G = {'x': 300, 'y': 500, 'z': 700}		#Zg>0
anchor_J = {'x': 200, 'y': 900, 'z': 900}		#Zj>0
sensor_S = {'x': 500, 'y': 500, 'z': 1000}
time_used_in_B = 0.5
time_used_in_C = 0.5
time_used_in_D = 0.5
time_used_in_E = 0.5
time_used_in_F = 0.5
time_used_in_G = 0.5
time_used_in_H = 0.5
time_used_in_I = 0.5
time_used_in_J = 0.5
time_interval1 = 1000
time_interval2 = 5000
time_interval3 = 1000
velocity = 1500
sa0 = 0
sb_a = 0
sc_a = 0
sd_a = 0
se_a = 0
sf_a = 0
sg_a = 0
sh_a = 0
si_a = 0
sj_a = 0
index_a = 0

def calculation(time_used_in_C, time_interval, velocity, time_used_in_B, anchor_A, anchor_B, anchor_C, anchor_D, sensor_S, time_used_in_D, sa0, ls, ms, ns):
    
    dt1_b = 0
    dt2_c = 0
    dt3_d = 0
    ls = 0
    ms = 0
    ns = 0
    comp = 0
    dtb(time_used_in_B, time_interval)
    dtc(time_used_in_C, time_interval, velocity, time_used_in_B, anchor_A, anchor_B, anchor_C, sensor_S	)
    dtd(time_used_in_C, time_interval, velocity, time_used_in_B, anchor_A, anchor_B, anchor_C, anchor_D	, sensor_S, time_used_in_D)
    dt1(time_used_in_C, time_interval, velocity, time_used_in_B, anchor_A, anchor_B, anchor_C, sensor_S	)
    dt2(time_used_in_C, time_interval, velocity, time_used_in_B, anchor_A, anchor_B, anchor_C, sensor_S	)
    dt3(time_used_in_C, time_interval, velocity, time_used_in_B, anchor_A, anchor_B, anchor_C, anchor_D	, sensor_S, time_used_in_D)

    with open('dt1.txt') as dt1_l:
       lines_dt1 = dt1_l.readlines()
    with open('dt2.txt') as dt2_l:
       lines_dt2 = dt2_l.readlines()
    with open('dt3.txt') as dt3_l:
       lines_dt3 = dt3_l.readlines()
    with open('dtb.txt') as dtb_l:
       lines_dtb = dtb_l.readlines()
    with open('dtc.txt') as dtc_l:
       lines_dtc = dtc_l.readlines()
    with open('dtd.txt') as dtd_l:
       lines_dtd = dtd_l.readlines()
	
    while lines_dt1:
        dt1_b = float(lines_dt1.pop()) - float(lines_dtb.pop())
        dt2_c = float(lines_dt2.pop()) - float(lines_dtc.pop())
        dt3_d = float(lines_dt3.pop()) - float(lines_dtd.pop())
        ab = get_distance(anchor_A, anchor_B)
        ac = get_distance(anchor_A, anchor_C)
        ad = get_distance(anchor_A, anchor_D)
        k1 = dt1_b*velocity - ab
        k2 = dt2_c*velocity - ac
        k3 = dt3_d*velocity - ad
        Ax = -k1/anchor_B['x']
        Bx = (anchor_B['x']**2-k1**2)/(2*anchor_B['x'])
        Ay = k1*anchor_C['x']/(anchor_B['x']*anchor_C['y']) - k2/anchor_C['y']
        By = (anchor_C['x']**2 + anchor_C['y']**2 - anchor_C['x']*anchor_B['x'] - k2**2 + anchor_C['x']*k1**2/anchor_B['x'])/(2*anchor_C['y'])
        Az = k1*anchor_D['x']/(anchor_B['x']*anchor_D['z']) - k3/anchor_D['z'] - anchor_D['y']*(k1*anchor_C['x']/anchor_B['x'] - k2)/(anchor_C['y']*anchor_D['z'])
        Bz = ((anchor_D['x']**2 + anchor_D['y']**2 + anchor_D['z']**2- anchor_D['x']*anchor_B['x'] - k3**2 + anchor_D['x']*k1**2/anchor_B['x'] - anchor_D['y']*anchor_C['x']**2/anchor_C['y']) + (-anchor_C['y']*anchor_D['y'] + anchor_B['x']*anchor_C['x']*anchor_D['y']/anchor_C['y'] - k1**2*anchor_C['x']*anchor_D['y']/(anchor_B['x']*anchor_C['y']) + k2**2*anchor_D['y']/anchor_C['y']))/(2*anchor_D['z'])
        alpha = Ax**2 + Ay**2 + Az**2 - 1
        beta = 2*(Ax*Bx + Ay*By + Az*Bz)
        gamma = Bx**2 + By**2 + Bz**2
        sa1 = (-beta-(beta**2-4*alpha*gamma)**0.5)/(2*alpha)
        sa2 = (-beta+(beta**2-4*alpha*gamma)**0.5)/(2*alpha)
        if isinstance(sa1,complex):
        	comp += 1
        else:
            if sa1 > 0:
                sa = sa1
            else:
                sa = sa2
        #print("(" + str(sa) + ")")
            xs = round(Ax*sa + Bx, 2)
            ys = round(Ay*sa + By, 2)
            zs = round(Az*sa + Bz, 2)
            sa0 += sa
            ls += ((xs-anchor_B['x'])**2+(ys-anchor_B['y'])**2+(zs-anchor_B['z'])**2)**0.5
            ms += ((xs-anchor_C['x'])**2+(ys-anchor_C['y'])**2+(zs-anchor_C['z'])**2)**0.5
            ns += ((xs-anchor_D['x'])**2+(ys-anchor_D['y'])**2+(zs-anchor_D['z'])**2)**0.5
    return (sa0, ls, ms, ns, comp)

def equations(guess):
    x, y, z, r = guess

    return (
        (x - anchor_A['x']) ** 2 + (y - anchor_A['y']) ** 2 + (z - anchor_A['z']) ** 2 - (d[0] - r) ** 2,
        (x - anchor_B['x']) ** 2 + (y - anchor_B['y']) ** 2 + (z - anchor_B['z']) ** 2 - (d[1] - r) ** 2,
        (x - anchor_C['x']) ** 2 + (y - anchor_C['y']) ** 2 + (z - anchor_C['z']) ** 2 - (d[2] - r) ** 2,
        (x - anchor_D['x']) ** 2 + (y - anchor_D['y']) ** 2 + (z - anchor_D['z']) ** 2 - (d[3] - r) ** 2,
        (x - anchor_E['x']) ** 2 + (y - anchor_E['y']) ** 2 + (z - anchor_E['z']) ** 2 - (d[4] - r) ** 2,
        (x - anchor_F['x']) ** 2 + (y - anchor_F['y']) ** 2 + (z - anchor_F['z']) ** 2 - (d[5] - r) ** 2,
        (x - anchor_G['x']) ** 2 + (y - anchor_G['y']) ** 2 + (z - anchor_G['z']) ** 2 - (d[6] - r) ** 2,
        (x - anchor_H['x']) ** 2 + (y - anchor_H['y']) ** 2 + (z - anchor_H['z']) ** 2 - (d[7] - r) ** 2,
        (x - anchor_I['x']) ** 2 + (y - anchor_I['y']) ** 2 + (z - anchor_I['z']) ** 2 - (d[8] - r) ** 2,
        (x - anchor_J['x']) ** 2 + (y - anchor_J['y']) ** 2 + (z - anchor_J['z']) ** 2 - (d[9] - r) ** 2,
    )

a = calculation(time_used_in_C, time_interval1, velocity, time_used_in_B, anchor_A, anchor_B, anchor_C, anchor_D, sensor_S, time_used_in_D, sa0, sb_a, sc_a, sd_a)	
b = calculation(time_used_in_F, time_interval2, velocity, time_used_in_E, anchor_A, anchor_E, anchor_F, anchor_G, sensor_S, time_used_in_G, sa0, se_a, sf_a, sg_a)
c = calculation(time_used_in_I, time_interval3, velocity, time_used_in_H, anchor_A, anchor_H, anchor_I, anchor_J, sensor_S, time_used_in_J, sa0, sh_a, si_a, sj_a)
index_a += (time_interval1+time_interval2+time_interval3-a[4]-b[4]-c[4])
d = ((a[0]+b[0]+c[0])/index_a,a[1]/(time_interval1-a[4]),a[2]/(time_interval1-a[4]),a[3]/(time_interval1-a[4]),b[1]/(time_interval2-b[4]),b[2]/(time_interval2-b[4]),b[3]/(time_interval2-b[4]),c[1]/(time_interval3-c[4]),c[2]/(time_interval3-c[4]),c[3]/(time_interval3-c[4]))
initial_guess = (0,0,0,0)
results = least_squares(equations, initial_guess)
print(results.x[0],results.x[1],results.x[2] )