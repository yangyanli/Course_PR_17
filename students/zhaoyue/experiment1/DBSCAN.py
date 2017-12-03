import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os

UNCLASSIFIED = False
NOISE = 0


def _dist(p, q):
    return math.sqrt(np.power(p - q, 2).sum())


def _eps_neighborhood(p, q, eps):
    return _dist(p, q) < eps


def _region_query(m, point_id, eps):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        if _eps_neighborhood(m[:, point_id], m[:, i], eps):
            seeds.append(i)
    return seeds


def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id

        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                                    classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True


def dbscan(m, eps, min_points):
    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        point = m[:, point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications


def main():
    parr = []
    carr = []

    with open(os.path.join('synthetic_data', 'flame.txt'), 'r') as f:
        s = f.readlines()
        for l in s:
            x, y, c = map(lambda x: float(x), l.strip().split(','))
            parr.append((x, y))
            carr.append(c)
    c = dbscan(np.matrix(parr).transpose(), 0.96, 5)
    l = max(c)
    print(l)
    color = [[random.random() for _ in range(3)] for _ in range(l + 1)]
    for i, j in zip(parr, c):
        m = '.'
        if j == 0:
            m = 'x'
        plt.scatter(*i, c=color[j], marker=m)
    plt.show()


if __name__ == '__main__':
    main()
