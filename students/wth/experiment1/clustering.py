# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, data, min_samples, eps=None):
        self.data = data
        self.min_samples = min_samples
        if eps is None:
            self.eps = self.decide_eps()
        else:
            self.eps = eps

    def decide_eps(self):
        dist = np.array([np.array([math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2) for y in self.data]) for x in self.data])
        neighbor = [np.sort(x) for x in dist]
        eps = np.average([x[self.min_samples] for x in neighbor])
        return eps

    def dbscan(self):
        labels = [0] * len(self.data)

        cluster = 0

        for point in range(0, len(self.data)):

            if not (labels[point] == 0):
                continue

            neighbor_points = self.region_query(point, self.eps)

            if len(neighbor_points) < self.min_samples:
                labels[point] = -1

            else:
                cluster += 1
                self.grow_cluster(labels, point, neighbor_points, cluster, self.eps, self.min_samples)

        return labels

    def grow_cluster(self, labels, seed_point, neighbor_points, cluster, eps, min_samples):
        labels[seed_point] = cluster

        i = 0
        while i < len(neighbor_points):

            p = neighbor_points[i]

            if labels[p] == -1:
                labels[p] = cluster

            elif labels[p] == 0:
                labels[p] = cluster
                p_neighbor_points = self.region_query(p, eps)

                if len(p_neighbor_points) >= min_samples:
                    neighbor_points = neighbor_points + p_neighbor_points

            i += 1

    def region_query(self, point, eps):

        neighbors = []

        for p in range(0, len(self.data)):

            if np.linalg.norm(self.data[point] - self.data[p]) < eps:
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
    np.random.shuffle(colors)
    colors[0] = "black"
    return colors


def run():

    colors = load_colors()

    #data_file_loc = "../../../experiment1/data/synthetic_data/Aggregation.txt"
    data_file_loc = "../../../experiment1/data/synthetic_data/Flame.txt"
    #data_file_loc = "../../../experiment1/data/synthetic_data/R15.txt"
    #data_file_loc = "../../../experiment1/data/synthetic_data/mix.txt"

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
    points_num = len(data["x"])

    dbs = DBSCAN(points, 12)
    labels = dbs.dbscan()

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Number of points: %d" % points_num)
    print("Estimated number of clusters: %d" % n_clusters_)

    point_colors = list(map(lambda x: colors[x+1], labels))

    fig, ax = plt.subplots()
    ax.scatter(data["x"], data["y"], color=point_colors)

    plt.show()


if __name__ == "__main__":
    run()
