import random
import numpy as np
def get_distance(a, b):
	distance = ((a['x']-b['x'])**2+(a['y']-b['y'])**2+(a['z']-b['z'])**2)**0.5
	return distance

def dtb(t, i):
	index = 0
	times = []
	while index < i:
		time = t
		time += time*np.random.normal(0,1e-4)
		times.append(time)
		index += 1
	with open('dtb.txt', 'w') as time_obj:
	    for time in times:
	        time_obj.write(str(time) + "\n")

def dtc(dtc, i, v, dtb, a, b, c, s):
	index = 0
	times = []
	while index < i:
		time = round((get_distance(a,b)+get_distance(b,c)-get_distance(a,c))/v + dtc + dtb, 6)
		time += time*np.random.normal(0,1e-4)
		times.append(time)
		index += 1
	with open('dtc.txt', 'w') as time_obj:
	    for time in times:
	        time_obj.write(str(time) + "\n")

def dtd(dtc, i, v, dtb, a, b, c, d, s, dtd):
	index = 0
	times = []
	while index < i:
		time = round((get_distance(a,b)+get_distance(b,c)+get_distance(c,d)-get_distance(a,d))/v +dtd + dtc + dtb, 6)
		time += time*np.random.normal(0,1e-4)
		times.append(time)
		index += 1
	with open('dtd.txt', 'w') as time_obj:
	    for time in times:
	        time_obj.write(str(time) + "\n")

def dt1(t, i, v, dtb, a, b, c, s):
	index = 0
	times = []
	while index < i:
		time = round((get_distance(a,b)+get_distance(b,s)-get_distance(a,s))/v + dtb, 6)
		time += time*np.random.normal(0,1e-4)
		times.append(time)
		index += 1
	with open('dt1.txt', 'w') as time_obj:
	    for time in times:
	        time_obj.write(str(time) + "\n")

def dt2(dtc, i, v, dtb, a, b, c, s):
	index = 0
	times = []
	while index < i:
		time = round((get_distance(a,b)+get_distance(b,c)+get_distance(c,s)-get_distance(a,s))/v + dtc + dtb, 6)
		time += time*np.random.normal(0,1e-4)
		times.append(time)
		index += 1
	with open('dt2.txt', 'w') as time_obj:
	    for time in times:
	        time_obj.write(str(time) + "\n")

def dt3(dtc, i, v, dtb, a, b, c, d, s, dtd):
	index = 0
	times = []
	while index < i:
		time = round((get_distance(a,b)+get_distance(b,c)+get_distance(c,d)+get_distance(d,s)-get_distance(a,s))/v + dtd + dtc + dtb, 6)
		time += time*np.random.normal(0,1e-4)
		times.append(time)
		index += 1
	with open('dt3.txt', 'w') as time_obj:
	    for time in times:
	        time_obj.write(str(time) + "\n")