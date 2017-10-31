# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def dbscan(data, eps, min_samples):
    labels = [0] * len(data)

    cluster = 0

    for point in range(0, len(data)):

        if not (labels[point] == 0):
            continue

        neighbor_points = region_query(data, point, eps)

        if len(neighbor_points) < min_samples:
            labels[point] = -1

        else:
            cluster += 1
            grow_cluster(data, labels, point, neighbor_points, cluster, eps, min_samples)

    return labels


def grow_cluster(data, labels, seed_point, neighbor_points, cluster, eps, min_samples):
    labels[seed_point] = cluster

    i = 0
    while i < len(neighbor_points):

        p = neighbor_points[i]

        if labels[p] == -1:
            labels[p] = cluster

        elif labels[p] == 0:
            labels[p] = cluster
            p_neighbor_points = region_query(data, p, eps)

            if len(p_neighbor_points) >= min_samples:
                neighbor_points = neighbor_points + p_neighbor_points

        i += 1


def region_query(data, point, eps):

    neighbors = []

    for p in range(0, len(data)):

        if np.linalg.norm(data[point] - data[p]) < eps:
            neighbors.append(p)

    return neighbors


def load_colors():
    color_file_loc = "rgb.txt"
    color_file = open(color_file_loc)

    colors = []

    for line in color_file.readlines():
        ls = line.split("\t")
        if ls[0][0] != "#" and ls[0][0].isalpha():
            colors.append("xkcd:"+ls[0])

    return colors


def run():

    colors = load_colors()

    data_file_loc = "../../../experiment1/data/synthetic_data/flame.txt"
    data_file = open(data_file_loc)
    data = {
        "x": [],
        "y": [],
        "t": [],
    }
    for line in data_file.readlines():
        x, y, t = line.split(",")
        x = float(x)
        y = float(y)
        t = int(t)
        data["x"].append(x)
        data["y"].append(y)
        data["t"].append(t)

    points = np.array([[data["x"][i], data["y"][i]] for i in range(0, len(data["x"]))])

    labels = dbscan(points, 1.5, 11)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)

    point_colors = list(map(lambda x: colors[x], labels))

    fig, ax = plt.subplots()
    ax.scatter(data["x"], data["y"], color=point_colors)

    plt.show()


if __name__ == "__main__":
    run()
