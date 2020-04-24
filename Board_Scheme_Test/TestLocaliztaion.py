import numpy as np
import utm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import umatrix_new as um
from array import *
from TDOALocalization import TDOALocalization


def lat_lon_to_utm(anchors_lat_lon_depth):
    # TODO: Interpolate UTM between to zones: Do not know how to do!

    m, n = anchors_lat_lon_depth.shape
    utm_locations = um.zeros(m, n)
    zones = []
    zone_letters = []
    for k in range(m):
        try:
            easting, northing, zone, zone_letter = utm.from_latlon(anchors_lat_lon_depth[k][0],
                                                                   anchors_lat_lon_depth[k][1])
        except Exception as error:
            print("Exception: " + str(error) + "\r\n")
        else:
            utm_locations[k] = array('f', [easting, northing, anchors_lat_lon_depth[k][2]])
            zones.append(zone)
            zone_letters.append(zone_letter)

    if len(zones) > 0 and all(elem == zones[0] for elem in zones) and len(zone_letters) and all(elem == zone_letters[0] for elem in zone_letters):
        return utm_locations, zones[0], zone_letters[0]
    else:
        return -1

# Set seed for Random number generator
np.random.seed(1234)

# Constants
DIST_SCALE = 1000.0
SOUND_SPEED = 1530  # meter/second
NUM_LOCATION_CYCLES = 3
NUM_BEACON_CYCLES = 1

# =================== Create Anchor Nodes =========================== #
NUM_ANCHORS = 6
if NUM_ANCHORS < 3:
    raise Exception("Number of Anchors can't be less than 3!")
Anchors_LL = um.zeros(NUM_ANCHORS, 3)
Anchors_LL[0] = [55.129525, -1.505859793, 0.0]
Anchors_LL[1] = [55.12956, -1.505839496, 0.0]
Anchors_LL[2] = [55.129595, -1.505859793, 0.0]
if NUM_ANCHORS >= 4:
    Anchors_LL[3] = [55.129595, -1.505900207, 0.0]
if NUM_ANCHORS >= 5:
    Anchors_LL[4] = [55.12956, -1.505920504, 0.0]
if NUM_ANCHORS >= 6:
    Anchors_LL[5] = [55.129525, -1.505900207, 0.0]
# Anchors_LL[0] = [55.129518, -1.506078, 0.0]
# Anchors_LL[1] = [55.129518, -1.505878, 0.0]
# Anchors_LL[2] = [55.129518, -1.505678, 0.0]
# if NUM_ANCHORS >= 4:
#     Anchors_LL[3] = [55.129635, -1.505678, 0.0]
# if NUM_ANCHORS >= 5:
#     Anchors_LL[4] = [55.129635, -1.505878, 0.0]
# if NUM_ANCHORS >= 6:
#     Anchors_LL[5] = [55.129635, -1.506078, 0.0]

# Convert LatLon to UTM coordinate system
Anchors_UTM, zone, zone_letter = lat_lon_to_utm(Anchors_LL)
Anchors = Anchors_UTM.transpose

anchor_locations = um.zeros(NUM_ANCHORS, 3)
bflag_anchor_locations = um.ones(1, NUM_ANCHORS)
anchor_locations = Anchors_UTM.copy()

# ===================== Create Sensor nodes ========================= #

NUM_SENSORS = 4
z_pos = array('f', [0.5170, 0.3656, 0.5157, 0.4661])
Sensors_LL = um.zeros(NUM_SENSORS, 3)
Sensors_LL[0] = array('f', [55.129574, -1.505886, z_pos[0]])
Sensors_LL[1] = array('f', [55.129560, -1.505878, z_pos[1]])
Sensors_LL[2] = array('f', [55.129547, -1.505871, z_pos[2]])
Sensors_LL[3] = array('f', [55.129560, -1.505878, z_pos[3]])

# Convert LatLon to UTM coordinate system
Sensors_UTM, zone, zone_letter = lat_lon_to_utm(Sensors_LL)
Sensors = Sensors_UTM.transpose

# ====================== Distance between Anchors: Known parameters============== #

AB = pow((pow(Anchors[0][0] - Anchors[0][1], 2) +
          pow(Anchors[1][0] - Anchors[1][1], 2) +
          pow(Anchors[2][0] - Anchors[2][1], 2)), 0.5)
AC = pow((pow(Anchors[0][0] - Anchors[0][2], 2) +
          pow(Anchors[1][0] - Anchors[1][2], 2) +
          pow(Anchors[2][0] - Anchors[2][2], 2)), 0.5)
BC = pow((pow(Anchors[0][1] - Anchors[0][2], 2) +
          pow(Anchors[1][1] - Anchors[1][2], 2) +
          pow(Anchors[2][1] - Anchors[2][2], 2)), 0.5)
if NUM_ANCHORS >= 4:
    AD = pow((pow(Anchors[0][0] - Anchors[0][3], 2) +
              pow(Anchors[1][0] - Anchors[1][3], 2) +
              pow(Anchors[2][0] - Anchors[2][3], 2)), 0.5)
    CD = pow((pow(Anchors[0][2] - Anchors[0][3], 2) +
              pow(Anchors[1][2] - Anchors[1][3], 2) +
              pow(Anchors[2][2] - Anchors[2][3], 2)), 0.5)
if NUM_ANCHORS >= 5:
    AE = pow((pow(Anchors[0][0] - Anchors[0][4], 2) +
              pow(Anchors[1][0] - Anchors[1][4], 2) +
              pow(Anchors[2][0] - Anchors[2][4], 2)), 0.5)
    DE = pow((pow(Anchors[0][3] - Anchors[0][4], 2) +
              pow(Anchors[1][3] - Anchors[1][4], 2) +
              pow(Anchors[2][3] - Anchors[2][4], 2)), 0.5)
if NUM_ANCHORS >= 6:
    AF = pow((pow(Anchors[0][0] - Anchors[0][5], 2) +
              pow(Anchors[1][0] - Anchors[1][5], 2) +
              pow(Anchors[2][0] - Anchors[2][5], 2)), 0.5)
    EF = pow((pow(Anchors[0][4] - Anchors[0][5], 2) +
              pow(Anchors[1][4] - Anchors[1][5], 2) +
              pow(Anchors[2][4] - Anchors[2][5], 2)), 0.5)
if NUM_ANCHORS >= 7:
    AG = pow((pow(Anchors[0][0] - Anchors[0][6], 2) +
              pow(Anchors[1][0] - Anchors[1][6], 2) +
              pow(Anchors[2][0] - Anchors[2][6], 2)), 0.5)
    FG = pow((pow(Anchors[0][5] - Anchors[0][6], 2) +
              pow(Anchors[1][5] - Anchors[1][6], 2) +
              pow(Anchors[2][5] - Anchors[2][6], 2)), 0.5)
if NUM_ANCHORS >= 8:
    AH = pow((pow(Anchors[0][0] - Anchors[0][7], 2) +
              pow(Anchors[1][0] - Anchors[1][7], 2) +
              pow(Anchors[2][0] - Anchors[2][7], 2)), 0.5)
    GH = pow((pow(Anchors[0][6] - Anchors[0][7], 2) +
              pow(Anchors[1][6] - Anchors[1][7], 2) +
              pow(Anchors[2][6] - Anchors[2][7], 2)), 0.5)


# ===== Distance Matrix: Distance between Sensor and Anchor pairs =========#
DistMat = um.zeros(NUM_SENSORS, NUM_ANCHORS)
a = []
for i in range(NUM_SENSORS):
    for j in range(NUM_ANCHORS):
        a.append(pow((pow(Sensors[0][i] - Anchors[0][j], 2) +
                      pow(Sensors[1][i] - Anchors[1][j], 2) +
                      pow(Sensors[2][i] - Anchors[2][j], 2)), 0.5))
    DistMat[i] = a
    a = []
"""
#=================== Phase-1: TDoA for Ranging ==============================#
#= Anchor A starts as a master, and sends the beacon signal every T seconds. The
sensor S and the anchor B, C, and D receive it at t1, tb, tc, and td respectively.
Anchor B replies A at time ̃tb ≥ tb with information Δtb =  ̃tb - tb, which reaches
to sensor S at t2. After receiving signal from both A and B, anchor C replies A
at ̃tc ≥ tc conveying Δtc = ̃tc - tc, which reaches sensor S at t3. After listening
anchor A, B, and C, anchor D replies A at ̃td ≥ td conveying Δtd = ̃td - td, which
reaches sensor S at t4.
#==============================================================================#
"""
# ================ We add i.i.d. noise to time measurements ===================#
σ = 0.000125
noise_type = "gaussian"  # "uniform" #
if noise_type == "uniform":
    lb = -σ*pow(3, 0.5)
    ub = σ*pow(3, 0.5)

# Array to store time stamps the sensor nodes
bflag_anchor_beacons = um.zeros(1, NUM_ANCHORS)
anchor_beacons = um.zeros(NUM_ANCHORS, 2)
bflag_anchor_beacons_tuple = []
anchor_beacons_tuple = []
for i in range(NUM_SENSORS):
    t1 = DistMat[i][0]/SOUND_SPEED
    tb = AB/SOUND_SPEED
    tc = AC/SOUND_SPEED
    if NUM_ANCHORS >= 4:
        td = AD/SOUND_SPEED
    if NUM_ANCHORS >= 5:
        te = AE/SOUND_SPEED
    if NUM_ANCHORS >= 6:
        tf = AF/SOUND_SPEED
    if NUM_ANCHORS >= 7:
        tg = AG/SOUND_SPEED
    if NUM_ANCHORS >= 8:
        th = AH/SOUND_SPEED
    if noise_type == "gaussian":
        t1 += σ*np.random.randn(1)[0]
        tb += σ*np.random.randn(1)[0]
        tc += σ*np.random.randn(1)[0]
        if NUM_ANCHORS >= 4:
            td += σ*np.random.randn(1)[0]
        if NUM_ANCHORS >= 5:
            te += σ*np.random.randn(1)[0]
        if NUM_ANCHORS >= 6:
            tf += σ*np.random.randn(1)[0]
        if NUM_ANCHORS >= 7:
            tg += σ*np.random.randn(1)[0]
        if NUM_ANCHORS >= 8:
            th += σ*np.random.randn(1)[0]
    elif noise_type == "uniform":
        t1 += (ub-lb)*np.random.rand(1)[0] + lb
        tb += (ub-lb)*np.random.rand(1)[0] + lb
        tc += (ub-lb)*np.random.rand(1)[0] + lb
        if NUM_ANCHORS >= 4:
            td += (ub-lb)*np.random.rand(1)[0] + lb
        if NUM_ANCHORS >= 5:
            te += (ub-lb)*np.random.rand(1)[0] + lb
        if NUM_ANCHORS >= 6:
            tf += (ub - lb) * np.random.rand(1)[0] + lb
        if NUM_ANCHORS >= 7:
            tg += (ub - lb) * np.random.rand(1)[0] + lb
        if NUM_ANCHORS >= 8:
            th += (ub - lb) * np.random.rand(1)[0] + lb
    delay = (1.75-1.0)*np.random.rand(1)[0] + 1.0  # Some random delay at Anchor
    tab = tb + delay
    delta_tb = tab - tb
    t2 = tab + DistMat[i][1]/SOUND_SPEED
    tabc = tab + BC/SOUND_SPEED
    if noise_type == "gaussian":
        t2 += σ*np.random.randn(1)[0]
        tabc += σ*np.random.randn(1)[0]
    elif noise_type == "uniform":
        t2 += (ub-lb)*np.random.randn(1)[0] + lb
        tabc += (ub-lb)*np.random.randn(1)[0] + lb
    delay = (2.0-1.0)*np.random.rand(1)[0] + 1.0  # Some random delay at Anchor
    tabc += delay
    delta_tc = tabc - tc
    t3 = tabc + DistMat[i][2]/SOUND_SPEED
    if noise_type == "gaussian":
        t3 += σ * np.random.randn(1)[0]
    elif noise_type == "uniform":
        t3 += (ub - lb) * np.random.rand(1)[0] + lb
    if NUM_ANCHORS >= 4:
        tabcd = tabc + CD/SOUND_SPEED
        if noise_type == "gaussian":
            tabcd += σ * np.random.randn(1)[0]
        elif noise_type == "uniform":
            tabcd += (ub - lb) * np.random.randn(1)[0] + lb
        delay = (2.5-1.0)*np.random.rand(1)[0] + 1.0  # Some random delay at Anchor
        tabcd += delay
        delta_td = tabcd - td
        t4 = tabcd + DistMat[i][3]/SOUND_SPEED
        if noise_type == "gaussian":
            t4 += σ*np.random.randn(1)[0]
        elif noise_type == "uniform":
            t4 += (ub-lb)*np.random.rand(1)[0] + lb
    if NUM_ANCHORS >= 5:
        tabcde = tabcd + DE/SOUND_SPEED
        if noise_type == "gaussian":
            tabcde += σ * np.random.randn(1)[0]
        elif noise_type == "uniform":
            tabcde += (ub - lb) * np.random.randn(1)[0] + lb
        delay = (2.2-1.0)*np.random.rand(1)[0] + 1.0
        tabcde += delay
        delta_te = tabcde - te
        t5 = tabcde + DistMat[i][4]/SOUND_SPEED
        if noise_type == "gaussian":
            t5 += σ * np.random.randn(1)[0]
        elif noise_type == "uniform":
            t5 += (ub-lb) * np.random.rand(1)[0] + lb
    if NUM_ANCHORS >= 6:
        tabcdef = tabcde + EF/SOUND_SPEED
        if noise_type == "gaussian":
            tabcdef += σ * np.random.randn(1)[0]
        elif noise_type == "uniform":
            tabcdef += (ub - lb) * np.random.randn(1)[0] + lb
        delay = (1.9-1.0) * np.random.rand(1)[0] + 1.2
        tabcdef += delay
        delta_tf = tabcdef - tf
        t6 = tabcdef + DistMat[i][5]/SOUND_SPEED
        if noise_type == "gaussian":
            t6 += σ * np.random.randn(1)[0]
        elif noise_type == "uniform":
            t6 += (ub - lb) * np.random.rand(1)[0] + lb
    if NUM_ANCHORS >= 7:
        tabcdefg = tabcdef + FG/SOUND_SPEED
        if noise_type == "gaussian":
            tabcdefg += σ * np.random.randn(1)[0]
        elif noise_type == "uniform":
            tabcdefg += (ub - lb) * np.random.randn(1)[0] + lb
        delay = (1.9-1.0) * np.random.rand(1)[0] + 1.2
        tabcdef += delay
        delta_tg = tabcdefg - tg
        t7 = tabcdefg + DistMat[i][6]/SOUND_SPEED
        if noise_type == "gaussian":
            t7 += σ * np.random.randn(1)[0]
        elif noise_type == "uniform":
            t7 += (ub - lb) * np.random.rand(1)[0] + lb
    if NUM_ANCHORS >= 8:
        tabcdefgh = tabcdefg + GH/SOUND_SPEED
        if noise_type == "gaussian":
            tabcdefgh += σ * np.random.randn(1)[0]
        elif noise_type == "uniform":
            tabcdefgh += (ub - lb) * np.random.randn(1)[0] + lb
        delay = (1.9-1.0) * np.random.rand(1)[0] + 1.2
        tabcdefgh += delay
        delta_th = tabcdefgh - th
        t8 = tabcdefgh + DistMat[i][7]/SOUND_SPEED
        if noise_type == "gaussian":
            t8 += σ * np.random.randn(1)[0]
        elif noise_type == "uniform":
            t8 += (ub - lb) * np.random.rand(1)[0] + lb
    bflag_anchor_beacons = um.ones(1, NUM_ANCHORS)
    bflag_anchor_beacons_tuple.append(bflag_anchor_beacons)
    if NUM_ANCHORS == 3:
        anchor_beacons = um.umatrix([t1, 0.0],
                                    [t2, delta_tb],
                                    [t3, delta_tc])
    elif NUM_ANCHORS == 4:
        anchor_beacons = um.umatrix([t1, 0.0],
                                    [t2, delta_tb],
                                    [t3, delta_tc],
                                    [t4, delta_td])
    elif NUM_ANCHORS == 5:
        anchor_beacons = um.umatrix([t1, 0.0],
                                    [t2, delta_tb],
                                    [t3, delta_tc],
                                    [t4, delta_td],
                                    [t5, delta_te])
    elif NUM_ANCHORS == 6:
        anchor_beacons = um.umatrix([t1, 0.0],
                                    [t2, delta_tb],
                                    [t3, delta_tc],
                                    [t4, delta_td],
                                    [t5, delta_te],
                                    [t6, delta_tf])
    elif NUM_ANCHORS == 7:
        anchor_beacons = um.umatrix([t1, 0.0],
                                    [t2, delta_tb],
                                    [t3, delta_tc],
                                    [t4, delta_td],
                                    [t5, delta_te],
                                    [t6, delta_tf],
                                    [t7, delta_tg])
    elif NUM_ANCHORS == 8:
        anchor_beacons = um.umatrix([t1, 0.0],
                                    [t2, delta_tb],
                                    [t3, delta_tc],
                                    [t4, delta_td],
                                    [t5, delta_te],
                                    [t6, delta_tf],
                                    [t7, delta_tg],
                                    [t8, delta_th])
    else:
        raise Exception("Maximum 6 Anchors at moment!")
    anchor_beacons_tuple.append(anchor_beacons)

# Estimate Location of the Sensors
estimated_locations = um.zeros(NUM_SENSORS, 3)
indices = []
for i in range(NUM_SENSORS):
    tdoa = TDOALocalization(SOUND_SPEED, z_pos[i])
    estimated_sol = tdoa.estimate_sensor_location(bflag_anchor_locations, anchor_locations,
                                                            bflag_anchor_beacons_tuple[i], anchor_beacons_tuple[i])
    if len(estimated_sol) > 0:
        print("Estimated Location: ", estimated_sol)
        estimated_locations[i] = estimated_sol
        indices.append(i)

estimated_locations = estimated_locations.transpose
# Convert Estimated UTM locations to LatLon
estimated_locations_LL = um.zeros(len(indices), 3)
a = []
for i in indices:
    try:
        lat, lon = utm.to_latlon(estimated_locations[0][i], estimated_locations[1][i], zone, zone_letter)
    except Exception as e:
        print("Exception: " + str(e) + "\r\n")
        estimated_location = array('f', [])
    else:
        a.append([lat, lon, estimated_locations[2][i]])
        estimated_locations_LL[i] = a[0]
        a = []

print("Estimated Locations in LatLon: \n", estimated_locations_LL)
print("Actual Locations in LatLon: \n", Sensors_LL)

# Calculate RMSE of estimated locations
SE = 0.0
count = 0
for i in indices:
    SE += (pow(Sensors[0][i] - estimated_locations[0][i], 2) +
           pow(Sensors[1][i] - estimated_locations[1][i], 2) +
           pow(Sensors[2][i] - estimated_locations[2][i], 2))
    count += 1

try:
    MSE = SE/count
except ZeroDivisionError:
    print("No sensor is localized")
    RMSE = 0.0
else:
    RMSE = pow(MSE, 0.5)

print("{:d} out of {:d} Sensors were localized with RMSE = {:0.4f}".format(count, NUM_SENSORS, RMSE))

# ================== Plot the Sensor Nodes in 3D ===============================#
plt.close('all')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection="3d")
for i in range(NUM_SENSORS):
    ax1.scatter3D(Sensors[0][i], Sensors[1][i], Sensors[2][i], c="g", s=15, marker="*")

for i in indices:
    ax1.scatter3D(estimated_locations[0][i], estimated_locations[1][i], estimated_locations[2][i], c="r", s=15, marker="o")
    ax1.text3D(estimated_locations[0][i], estimated_locations[1][i], estimated_locations[2][i], str(i), fontsize=15)

anchor_name = ["A", "B", "C", "D", "E", "F", "G", "H"]
for i in range(NUM_ANCHORS):
    ax1.scatter3D(Anchors[0][i], Anchors[1][i], Anchors[2][i], c="b", s=25, marker="o")
    ax1.text3D(Anchors[0][i], Anchors[1][i], Anchors[2][i], anchor_name[i], fontsize=15)

# ax1.view_init(-145, -20)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
