import umatrix_new as um
import array as ua

class TDOALocalization:
    def __init__(self, sound_speed, depth):
        self.sound_speed = sound_speed
        self.depth = depth

    @staticmethod
    def extract_tdoa_information(bflag_anchor_locations, anchor_locations, bflag_anchor_beacons, anchor_beacons):
        final_anchor_locations = ua.array('f', [])
        final_anchor_beacons = ua.array('f', [])
        bflag_anchor_locations_flag = 1
        bflag_anchor_beacons_flag = 1

        f_l_shape = bflag_anchor_locations.shape
        for i in range(f_l_shape[0]):
            for j in range(f_l_shape[1]):
                if bflag_anchor_locations[i][j] == 0:
                    bflag_anchor_locations_flag = 0

        if bflag_anchor_locations_flag:
            final_anchor_locations = anchor_locations.copy()

        f_b_shape = bflag_anchor_beacons.shape
        for i in range(f_b_shape[0]):
            for j in range(f_b_shape[1]):
                if bflag_anchor_beacons[i][j] == 0:
                    bflag_anchor_beacons_flag = 0

        if bflag_anchor_beacons_flag:
            final_anchor_beacons = anchor_beacons.copy()

        return final_anchor_locations, final_anchor_beacons

    @staticmethod
    def calculate_ranges_and_timediffs(anchor_locations, anchor_beacons):
        if anchor_locations.size() != 0 and anchor_beacons.size() != 0:
            n, p = anchor_locations.shape
            ranges = um.zeros(1, n - 1)
            for i in range(1, n):
                ranges[0][i - 1] = pow((pow(anchor_locations[0][0] - anchor_locations[i][0], 2) +
                                        pow(anchor_locations[0][1] - anchor_locations[i][1], 2) +
                                        pow(anchor_locations[0][2] - anchor_locations[i][2], 2)), 0.5)

            n, p = anchor_beacons.shape
            timediffs = um.zeros(n - 1, p)
            for i in range(1, n):
                timediffs[i - 1] = [anchor_beacons[i][0] - anchor_beacons[0][0],
                                    anchor_beacons[i][1]]

            return ranges, timediffs
        else:
            return ua.array('f', []), ua.array('f', [])

    @staticmethod
    def calculate_k(soundspeed, anchor_ranges, anchor_timediffs):
        if anchor_ranges.size() > 0 and anchor_timediffs.size() > 0:
            n, p = anchor_timediffs.shape
            K = um.zeros(1, n)
            for j in range(n):
                K[0][j] = soundspeed * (anchor_timediffs[j][1] - anchor_timediffs[j][0]) + anchor_ranges[0][j]
            return K
        else:
            return ua.array('f', [])

    @staticmethod
    def perform_tdoa_multilateration(anchor_locations, sensor_depth, K):
        sol = ua.array('f', [])
        case_num = None
        if anchor_locations.size() > 0 and K.size() > 0:
            n = K.size()
            matM = um.zeros(n, 2)
            for i in range(n):
                matM[i] = [2.0 * (anchor_locations[0][0] - anchor_locations[i + 1][0]),
                           2.0 * (anchor_locations[0][1] - anchor_locations[i + 1][1])]

            vecC = 2.0 * K
            vecD = um.zeros(1, n)
            for i in range(n):
                vecD[0][i] = pow(anchor_locations[i + 1][0], 2) + pow(anchor_locations[i + 1][1], 2) + \
                             pow(anchor_locations[i + 1][2], 2) - pow(anchor_locations[0][0], 2) - \
                             pow(anchor_locations[0][1], 2) - pow(anchor_locations[0][2], 2) - \
                             K[0][i]**2 + 2 * sensor_depth * (anchor_locations[0][2] - anchor_locations[i + 1][2])

            matMinv = ((matM.transpose * matM).inverse) * (matM.transpose)
            vecA = -1 * matMinv * (vecC.transpose)
            vecB = -1 * matMinv * (vecD.transpose)
            alpha = (vecA.transpose * vecA)[0][0] - 1.0
            vec = um.umatrix([anchor_locations[0][0], anchor_locations[0][1]])
            beta = (2.0 * (vecA.transpose * (vecB - vec.transpose)))[0][0]
            gamma = ((vecB - vec.transpose).transpose * (vecB - vec.transpose))[0][0] + \
                    (sensor_depth - anchor_locations[0][2]) ** 2
            delta = beta**2 - 4.0*alpha*gamma
            if delta < 0.0:
                sol = ua.array('f', [])

            elif abs(delta - 0.0) < 1E-5 and beta < 0.0:
                root = -beta / (2.0 * alpha)
                sol = root * vecA + vecB

            elif abs(alpha - 0.0) < 1E-5 and beta < 0.0:
                root = -gamma / beta
                sol = root * vecA + vecB

            elif delta > 0.0 and not abs(alpha - 0.0) < 1E-5:
                sqrt_delta = pow(delta, 0.5)
                if alpha < 0.0:
                    root = (-beta - sqrt_delta) / (2.0 * alpha)
                    sol = root * vecA + vecB

                else:
                    root1 = (-beta - sqrt_delta) / (2 * alpha)
                    root2 = (-beta + sqrt_delta) / (2 * alpha)
                    if root2 < 0.0 < root1:
                        sol = root1 * vecA + vecB

                    elif root1 < 0.0 < root2:
                        sol = root2 * vecA + vecB

                    elif root1 > 0.0 and root2 > 0.0:
                        sol1 = root1 * vecA + vecB
                        sol2 = root2 * vecA + vecB
                        dist1 = 0.0
                        dist2 = 0.0
                        a_l_shape = anchor_locations.shape
                        for j in range(a_l_shape[0]):
                            dist1 += pow((pow(anchor_locations[j][0] - sol1[0][0], 2) +
                                          pow(anchor_locations[j][1] - sol1[1][0], 2)), 0.5)
                            dist2 += pow((pow(anchor_locations[j][0] - sol2[0][0], 2) +
                                          pow(anchor_locations[j][1] - sol2[1][0], 2)), 0.5)

                        if dist1 < dist2:
                            sol = sol1

                        elif dist1 > dist2:
                            sol = sol2

        if sol.size() > 0:
            return ua.array('f', [sol[0][0], sol[1][0], sensor_depth])
        else:
            return ua.array('f', [])

    def estimate_sensor_location(self,
                                 bflag_anchor_locations, anchor_locations, bflag_anchor_beacons, anchor_beacons):
        anchor_locations, anchor_beacons = self.extract_tdoa_information(bflag_anchor_locations, anchor_locations,
                                                                         bflag_anchor_beacons, anchor_beacons)

        ranges, timediffs = self.calculate_ranges_and_timediffs(anchor_locations, anchor_beacons)

        K = self.calculate_k(self.sound_speed, ranges, timediffs)

        estimated_location = self.perform_tdoa_multilateration(anchor_locations, self.depth, K)

        return estimated_location
