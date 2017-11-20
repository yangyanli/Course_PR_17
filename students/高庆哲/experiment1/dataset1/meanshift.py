import math
import sys
import numpy as np

MIN_DISTANCE = 0.000001


def gaussian_kernel(distance, bandwidth):
    euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((euclidean_distance / bandwidth))**2)
    return val


class MeanShift(object):
    def __init__(self, kernel = gaussian_kernel):

        if kernel == 'multivariate_gaussian':
            kernel = multivariate_gaussian_kernel

        self.kernel = kernel

    def cluster(self, points, kernel_bandwidth, iteration_callback = None):
        if(iteration_callback):
            iteration_callback(points, 0)
        shift_points = np.array(points)
        max_min_dist = 1
        iteration_number = 0

        still_shifting = [True] * points.shape[0]
        while max_min_dist > MIN_DISTANCE:
            # print max_min_dist
            max_min_dist = 0
            iteration_number += 1
            for i in range(0, len(shift_points)):
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new
                p_new = self._shift_point(p_new, points, kernel_bandwidth)
                dist = euclidean_dist(p_new, p_new_start)
                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                shift_points[i] = p_new
            if iteration_callback:
                iteration_callback(shift_points, iteration_number)
        point_grouper = PointGrouper()
        group_assignments = point_grouper.group_points(shift_points.tolist())
        return MeanShiftResult(points, shift_points, group_assignments)

    def _shift_point(self, point, points, kernel_bandwidth):
        # from http://en.wikipedia.org/wiki/Mean-shift
        points = np.array(points)

        # numerator
        point_weights = self.kernel(point-points, kernel_bandwidth)
        tiled_weights = np.tile(point_weights, [len(point), 1])
        # denominator
        denominator = sum(point_weights)
        shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
        return shifted_point

        # ***************************************************************************
        # ** The above vectorized code is equivalent to the unrolled version below **
        # ***************************************************************************
        # shift_x = float(0)
        # shift_y = float(0)
        # scale_factor = float(0)
        # for p_temp in points:
        #     # numerator
        #     dist = euclidean_dist(point, p_temp)
        #     weight = self.kernel(dist, kernel_bandwidth)
        #     shift_x += p_temp[0] * weight
        #     shift_y += p_temp[1] * weight
        #     # denominator
        #     scale_factor += weight
        # shift_x = shift_x / scale_factor
        # shift_y = shift_y / scale_factor
        # return [shift_x, shift_y]

class MeanShiftResult:
    def __init__(self, original_points, shifted_points, cluster_ids):
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.cluster_ids = cluster_ids




GROUP_DISTANCE_TOLERANCE = .1


class PointGrouper(object):
    def group_points(self, points):
        group_assignment = []
        groups = []
        group_index = 0
        index = 0
        for point in points:
            nearest_group_index = self._determine_nearest_group(point, groups)
            if nearest_group_index == None:
                # create new group
                groups.append([point])
                group_assignment.append(group_index)
                group_index += 1
            else:
                group_assignment.append(nearest_group_index)
                groups[nearest_group_index].append(point)
            index += 1
        return np.array(group_assignment)

    def _determine_nearest_group(self, point, groups):
        nearest_group_index = None
        index = 0
        for group in groups:
            distance_to_group = self._distance_to_group(point, group)
            if distance_to_group < GROUP_DISTANCE_TOLERANCE:
                nearest_group_index = index
            index += 1
        return nearest_group_index

    def _distance_to_group(self, point, group):
        min_distance = sys.float_info.max
        for pt in group:
            dist = euclidean_dist(point, pt)
            if dist < min_distance:
                min_distance = dist
        return min_distance




def euclidean_dist(pointA, pointB):
    if(len(pointA) != len(pointB)):
        raise Exception("expected point dimensionality to match")
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    return math.sqrt(total)




def multivariate_gaussian_kernel(distances, bandwidths):

    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)

    # Covariance matrix
    cov = np.multiply(np.power(bandwidths,2), np.eye(dim))

    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val = (1 / np.power((2 * math.pi), (dim/2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)

    return val