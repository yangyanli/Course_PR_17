import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys
import os

MIN_DISTANCE = 0.0001
GROUP_DISTANCE_TOLERANCE = .1


def euclidean_dist(pointA, pointB):
    if len(pointA) != len(pointB):
        raise Exception("expected point dimensionality to match")
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension]) ** 2
    return math.sqrt(total)


def gaussian_kernel(distance, bandwidth):
    euclidean_distance = np.sqrt((distance ** 2).sum(axis=1))
    val = (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(-0.5 * (euclidean_distance / bandwidth) ** 2)
    return val


def multivariate_gaussian_kernel(distances, bandwidths):
    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)

    # Covariance matrix
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))

    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val = (1 / np.power((2 * math.pi), (dim / 2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)

    return val


def group_points(points):
    group_assignment = []
    groups = []
    group_index = 0
    index = 0
    for point in points:
        nearest_group_index = determine_nearest_group(point, groups)
        if nearest_group_index is None:
            groups.append([point])
            group_assignment.append(group_index)
            group_index += 1
        else:
            group_assignment.append(nearest_group_index)
            groups[nearest_group_index].append(point)
        index += 1
    return np.array(group_assignment)


def determine_nearest_group(point, groups):
    nearest_group_index = None
    index = 0
    for group in groups:
        _distance_to_group = distance_to_group(point, group)
        if _distance_to_group < GROUP_DISTANCE_TOLERANCE:
            nearest_group_index = index
        index += 1
    return nearest_group_index


def distance_to_group(point, group):
    min_distance = sys.float_info.max
    for pt in group:
        dist = euclidean_dist(point, pt)
        if dist < min_distance:
            min_distance = dist
    return min_distance


def cluster(points, kernel_bandwidth, kernel, iteration_callback=None):
    if iteration_callback:
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
            p_new = shift_point(p_new, points, kernel_bandwidth, kernel)
            dist = euclidean_dist(p_new, p_new_start)
            if dist > max_min_dist:
                max_min_dist = dist
            if dist < MIN_DISTANCE:
                still_shifting[i] = False
            shift_points[i] = p_new
        if iteration_callback:
            iteration_callback(shift_points, iteration_number)
    group_assignments = group_points(shift_points.tolist())
    return points, shift_points, group_assignments


def shift_point(point, points, kernel_bandwidth, kernel):
    # from http://en.wikipedia.org/wiki/Mean-shift
    points = np.array(points)

    # numerator
    point_weights = kernel(point - points, kernel_bandwidth)
    tiled_weights = np.tile(point_weights, [len(point), 1])
    # denominator
    denominator = sum(point_weights)
    shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
    return shifted_point


def main():
    parr = []
    carr = []

    with open(os.path.join('synthetic_data', 'mix.txt'), 'r') as f:
        s = f.readlines()
        for l in s:
            x, y, c = map(lambda x: float(x), l.strip().split(','))
            parr.append((x, y))
            carr.append(c)
    p, s, c = cluster(np.array(parr), 2, gaussian_kernel)
    l = max(c) + 1
    print(l)
    color = [[random.random() for _ in range(3)] for _ in range(l)]
    for i, j in zip(p, c):
        plt.scatter(*i, c=color[j], marker='.')
    plt.show()


if __name__ == '__main__':
    main()
