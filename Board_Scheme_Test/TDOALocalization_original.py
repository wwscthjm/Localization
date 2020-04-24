import math
import numpy as np
import umatrix as um
from array import *


class TDOALocalization:
    def __init__(self, soundspeed, depth):
        self.soundspeed = soundspeed
        self.depth = depth

    @staticmethod
    def extract_tdoa_information(bflag_anchor_locations, anchor_locations, bflag_anchor_beacons, anchor_beacons):
        """
        extract_tdoa_information check if anchor's location and beacon signal are available and extract them into
        numpy arrays
        :return: two arrays containing anchor's locations and beacon signal information, otherwise two empty arrays
        """
        final_anchor_locations = array('f', [])
        final_anchor_beacons = array('f', [])

        # Extract the Anchor Locations
        try:
            if (bflag_anchor_locations.ndim == 2) and (anchor_locations.ndim == 3):
                m, n = bflag_anchor_locations.shape
                bflags = [bflag_anchor_locations[:, i].any() for i in range(n)]
                if bflags.count(True) == n:
                    # All the anchor's locations are received at least once in whole tdoa cycle
                    # Calculate the Mean Locations of each Anchors
                    final_anchor_locations = np.sum(anchor_locations, axis=0)
                    divisor = np.sum(bflag_anchor_locations, axis=0)
                    for i in range(3):
                        final_anchor_locations[:, i] = final_anchor_locations[:, i] / divisor
                else:
                    pass
            elif (bflag_anchor_locations.ndim == 1) and (anchor_locations.ndim == 2):
                if bflag_anchor_locations.all():
                    final_anchor_locations = anchor_locations
                else:
                    pass
            else:
                pass
        except Exception as e:
            print("Couldn't extract Anchor Locations!")
            print("Exception: " + str(e) + "\r\n")

        try:
            # Extracts the Beacon Signal
            if (bflag_anchor_beacons.ndim == 2) and (anchor_beacons.ndim == 3):
                if bflag_anchor_beacons.all():
                    final_anchor_beacons = anchor_beacons
                elif bflag_anchor_beacons.any():
                    indices = []
                    m, n = bflag_anchor_beacons.shape
                    for i in range(m):
                        if bflag_anchor_beacons[i, :].all():
                            indices.append(i)

                    m, n, p = anchor_beacons.shape
                    k = len(indices)
                    if k > 0:
                        final_anchor_beacons = np.zeros((k, n, p), dtype=float)
                        for i in range(k):
                            final_anchor_beacons[i, :] = anchor_beacons[indices[i], :]
                    else:
                        pass
            elif (bflag_anchor_beacons.ndim == 1) and (anchor_beacons.ndim == 2):
                if bflag_anchor_beacons.all():
                    final_anchor_beacons = anchor_beacons
                else:
                    pass
            else:
                pass
        except Exception as e:
            print("Couldn't extract Anchor Beacons!")
            print("Exception: " + str(e) + "\r\n")

        return final_anchor_locations, final_anchor_beacons

    @staticmethod
    def calculate_ranges_and_timediffs(anchor_locations, anchor_beacons):
        """

        :param anchor_locations: should be nx3 array where n is number of anchors
        :param anchor_beacons: should be mxnx2 or nx2 array where m is number cycles, n is number of anchors
        :return: anchor_ranges, timestamps
        """
        if anchor_locations.size != 0 and anchor_beacons.size != 0:
            # calculate range of slave anchors from master anchor
            n, p = anchor_locations.shape
            ranges = np.zeros(n - 1, dtype=float)
            for i in range(1, n):
                ranges[i - 1] = np.linalg.norm(anchor_locations[0] - anchor_locations[i])

            # calculate the timestamps from anchor beacon arrival instances and delays
            if anchor_beacons.ndim == 3:
                m, n, p = anchor_beacons.shape
                timediffs = np.zeros((m, n - 1, p), dtype=float)
                for i in range(m):
                    for j in range(1, n):
                        timediffs[i, j - 1] = np.array([(anchor_beacons[i, j, 0] - anchor_beacons[i, 0, 0]), anchor_beacons[i, j, 1]])
            else:
                n, p = anchor_beacons.shape
                timediffs = np.zeros((n - 1, p), dtype=float)
                for j in range(1, n):
                    timediffs[j - 1] = np.array([(anchor_beacons[j, 0] - anchor_beacons[0, 0]), anchor_beacons[j, 1]])

            return ranges, timediffs
        else:
            return np.array([]), np.array([])

    @staticmethod
    def calculate_k(soundspeed, anchor_ranges, anchor_timediffs, noise_type="uniform"):
        if anchor_ranges.size > 0 and anchor_timediffs.size > 0:
            if anchor_timediffs.ndim == 3:
                m, n, p = anchor_timediffs.shape
                K = np.zeros(n, dtype=float)
                if noise_type == "gaussian":
                    for j in range(n):
                        for i in range(m):
                            K[j] += anchor_timediffs[i, j, 1] - anchor_timediffs[i, j, 0]
                        K[j] = (soundspeed * K[j] / m) + anchor_ranges[j]
                elif noise_type == "uniform":
                    KS = np.zeros((n, m), dtype=float)
                    for j in range(n):
                        for i in range(m):
                            KS[j, i] = anchor_timediffs[i, j, 1] - anchor_timediffs[i, j, 0]

                        K[j] = np.median(KS[j, :])
                        K[j] = (soundspeed * K[j]) + anchor_ranges[j]
            else:
                n, p = anchor_timediffs.shape
                K = np.zeros(n, dtype=float)
                for j in range(n):
                    K[j] = soundspeed * (anchor_timediffs[j, 1] - anchor_timediffs[j, 0]) + anchor_ranges[j]

            return K
        else:
            return np.array([])

    @staticmethod
    def perform_tdoa_multilateration(anchor_locations, sensor_depth, K):
        """

        :return: estimated location as an array
        """
        sol = array('f', [])
        case_num = None
        if anchor_locations.size > 0 and K.size > 0:
            n = K.size
            matM = np.zeros((n, 2), dtype=float)
            for i in range(n):
                matM[i, :] = 2.0 * np.array([(anchor_locations[0, 0] - anchor_locations[i + 1, 0]),
                                             (anchor_locations[0, 1] - anchor_locations[i + 1, 1])])
            vecC = 2.0 * K
            vecD = np.zeros(n, dtype=float)
            for i in range(n):
                vecD[i] = np.linalg.norm(anchor_locations[i + 1])**2 - np.linalg.norm(anchor_locations[0])**2 \
                          - K[i]**2 + 2.0*(anchor_locations[0, 2] - anchor_locations[i + 1, 2]) * sensor_depth

            try:
                #vecA = -np.linalg.lstsq(matM, vecC, rcond=None)[0]
                #vecB = -np.linalg.lstsq(matM, vecD, rcond=None)[0]
                matMinv = np.matmul((np.linalg.inv(np.matmul(np.transpose(matM), matM))), np.transpose(matM))
                vecA = -np.dot(matMinv, vecC)
                vecB = -np.dot(matMinv, vecD)
            except Exception as e:
                print("Exception: " + str(e) + "\r\n")
                sol = np.array([])
                case_num = 0
                # print("I'm in Case-{:d} \n".format(case_num))
            else:
                # solve the quadratic equations
                alpha = np.dot(vecA, vecA) - 1.0
                beta = 2.0 * np.dot(vecA, (vecB - anchor_locations[0][0:2]))
                gamma = np.dot((vecB - anchor_locations[0][0:2]), (vecB - anchor_locations[0][0:2])) + (sensor_depth - anchor_locations[0][2]) ** 2
                delta = beta**2 - 4.0*alpha*gamma

                # print("alpha={:0.4f}, beta={:0.4f}, gamma={:0.4f}, delta={:0.4f}".format(alpha,beta,gamma,delta))

                if delta < 0.0:
                    sol = np.array([])  # No feasible solution
                    case_num = 1
                    # print("I'm in Case-{:d} \n".format(case_num))
                elif math.isclose(delta, 0.0, abs_tol=1E-5) and beta < 0.0:
                    root = -beta / (2.0 * alpha)
                    sol = root * vecA + vecB
                    case_num = 2
                    # print("I'm in Case-{:d} \n".format(case_num))
                elif math.isclose(alpha, 0.0, abs_tol=1E-5) and beta < 0.0:
                    root = -gamma / beta
                    sol = root * vecA + vecB
                    case_num = 3
                    # print("I'm in Case-{:d} \n".format(case_num))
                elif delta > 0.0 and not math.isclose(alpha, 0.0, abs_tol=1E-5):
                    # print("delta > 0 and alpha != 0")
                    sqrt_delta = np.sqrt(delta)
                    if alpha < 0.0:
                        root = (-beta - sqrt_delta) / (2.0 * alpha)
                        sol = root * vecA + vecB
                        case_num = 4
                        # print("I'm in Case-{:d} \n".format(case_num))
                    else:
                        root1 = (-beta - sqrt_delta) / (2 * alpha)
                        root2 = (-beta + sqrt_delta) / (2 * alpha)
                        # print("Root1={:0.4f} and Root2={:0.4f}".format(root1, root2))
                        if root2 < 0.0 < root1:
                            sol = root1 * vecA + vecB
                            case_num = 5
                            # print("I'm in Case-{:d} \n".format(case_num))
                        elif root1 < 0.0 < root2:
                            sol = root2 * vecA + vecB
                            case_num = 6
                            # print("I'm in Case-{:d} \n".format(case_num))
                        elif root1 > 0.0 and root2 > 0.0:
                            sol1 = root1 * vecA + vecB
                            sol2 = root2 * vecA + vecB
                            dist1 = 0.0
                            dist2 = 0.0
                            for j in range(len(anchor_locations)):
                                dist1 += np.linalg.norm(anchor_locations[j][0:2] - sol1)
                                dist2 += np.linalg.norm(anchor_locations[j][0:2] - sol2)
                            if dist1 < dist2:
                                sol = sol1
                                case_num = 7
                                # print("I'm in Case-{:d} \n".format(case_num))
                            elif dist1 > dist2:
                                sol = sol2
                                case_num = 8
                                # print("I'm in Case-{:d} \n".format(case_num))
                            else:  # Two solutions
                                sol = np.array([])
                                case_num = 9
                                # print("I'm in Case-{:d} \n".format(case_num))
                        else:  # Both roots are negative
                            sol = np.array([])
                            case_num = 10
                            # print("I'm in Case-{:d} \n".format(case_num))
                else:
                    sol = np.array([])
                    case_num = 11
                    # print("I'm in Case-{:d} \n".format(case_num))

        if sol.size > 0:
            return case_num, np.array([sol[0], sol[1], sensor_depth])
        else:
            return case_num, np.array([])

    def estimate_sensor_location(self, bflag_anchor_locations, anchor_locations, bflag_anchor_beacons, anchor_beacons, noise_type='uniform'):
        # Step-1 Extract Useful Information from Collected Information
        anchor_locations, anchor_beacons = self.extract_tdoa_information(bflag_anchor_locations, anchor_locations, bflag_anchor_beacons, anchor_beacons)

        # Step-2 Calculate Slave Anchor ranges and Time Differences with respect to Master Anchor
        ranges, timediffs = self.calculate_ranges_and_timediffs(anchor_locations, anchor_beacons)

        #print("TimeStamps = ", timediffs)

        # Step-3 Calculate pre-requisite for Multilateration
        K = self.calculate_k(self.soundspeed, ranges, timediffs, noise_type=noise_type)
        #print("K = ", K)

        # Step-4 Perform Multilateration to estimate sensor location
        estimated_location = self.perform_tdoa_multilateration(anchor_locations, self.depth, K)

        return estimated_location
