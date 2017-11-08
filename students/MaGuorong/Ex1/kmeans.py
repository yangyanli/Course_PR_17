# -*- coding: utf-8 -*-

import numpy as np
import math


class k_means:
    def __init__(self, points_list, seeds, k):
        self.centroids = seeds
        self.cluster_count = k
        self.check_seeds()
        self.points_list = points_list
        self.points_count = len(points_list)
        self.cluster_size = math.floor(self.points_count / k)
        self.assign_initial_clusters()
        self.compute_clusters()

    def check_seeds(self):
        if len(self.centroids) != self.cluster_count:
            raise ValueError("Invalid seed length!")


    def assign_initial_clusters(self):
        self.clusters = [[] for i in range(self.cluster_count)]
        almost_full_clusters = []

        for j in range(self.points_count):
            if len(almost_full_clusters) == self.cluster_count:
                self.overflow_clusters(j)
                break
            index = self.nearest_centroid(self.points_list[j], almost_full_clusters)
            self.clusters[index].append(j)
            if self.is_almost_full(index):
                almost_full_clusters.append(index)

    """Finds nearest centroid to given point, denoted by its index in the points_list"""

    def nearest_centroid(self, point, full_clusters):
        best_centroid = -1
        min_distance = np.inf

        for i in range(self.cluster_count):
            distance = self.euclidean_distance(self.centroids[i], point)
            if distance < min_distance and i not in full_clusters:
                best_centroid = i
                min_distance = distance

        return best_centroid

    def is_almost_full(self, index):
        return len(self.clusters[index]) == self.cluster_size

    def is_full(self, index):
        return len(self.clusters[index]) == self.cluster_size + 1

    def overflow_clusters(self, index):
        full_clusters = []

        for i in range(index, self.points_count):
            index = self.nearest_centroid(self.points_list[i], full_clusters)
            self.clusters[index].append(i)
            if self.is_full(index):
                full_clusters.append(index)

    def euclidean_distance(self, point1, point2):
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)

        return np.linalg.norm(point2 - point1)

    def compute_clusters(self):
        iteration_count = 50

        for k in range(iteration_count):
            self.compute_new_centroids()
            self.iteration()

    def nearest_centroids(self, point):
        current_cluster = self.cluster_number(point)
        current_centroids = self.centroids[current_cluster]
        current_distance = self.euclidean_distance(current_centroids, self.points_list[point])
        better_centroids = []

        for j in range(len(self.centroids)):
            if self.euclidean_distance(self.centroids[j], self.points_list[point]) < current_distance:
                better_centroids.append(j)

        return better_centroids

    def iteration(self):
        for i in range(len(self.points_list)):
            closest_clusters = self.nearest_centroids(i)
            coords = self.find_index(i)
            if closest_clusters:
                best_point = self.best_swap_candidate(i, closest_clusters)
                if best_point != -1:
                    swap_cluster = self.cluster_number(best_point)

                    if len(self.clusters[swap_cluster]) < len(self.clusters[coords[0]]):
                        swap_point = self.clusters[coords[0]].pop(coords[1])
                        self.clusters[swap_cluster].append(swap_point)
                    else:
                        self.clusters[coords[0]].remove(i)
                        self.clusters[swap_cluster].remove(best_point)
                        self.clusters[swap_cluster].append(i)
                        self.clusters[coords[0]].append(best_point)

    def best_swap_candidate(self, original_point, candidate_clusters):
        best_point_candidate = -1
        best_change = 0
        for cluster_numbers in candidate_clusters:
            for point in self.clusters[cluster_numbers]:
                change = self.error_change(original_point, point)
                if change > best_change:
                    best_change = change
                    best_point_candidate = point

        return best_point_candidate

    def find_index(self, point):
        for i in range(self.cluster_count):
            if point in self.clusters[i]:
                coords = [i, self.clusters[i].index(point)]
                return coords

    def error_change(self, point1, point2):
        cluster_one = self.cluster_number(point1)
        cluster_two = self.cluster_number(point2)
        pre_error = self.euclidean_distance(self.points_list[point1], self.centroids[cluster_one]) + \
                    self.euclidean_distance(self.points_list[point2], self.centroids[cluster_two])
        new_error = self.euclidean_distance(self.points_list[point1], self.centroids[cluster_two]) + \
                    self.euclidean_distance(self.points_list[point2], self.centroids[cluster_one])
        return pre_error - new_error

    def cluster_number(self, point):
        index = -1

        for cluster in self.clusters:
            if point in cluster:
                index = self.clusters.index(cluster)

        return index

    def compute_new_centroids(self):
        new_centroids = []

        for i in range(self.cluster_count):
            l = []
            for point_index in self.clusters[i]:
                l.append(self.points_list[point_index])
            l = np.asarray(l)
            new_centroids.append(np.mean(l, axis=0).tolist())

        self.centroids = new_centroids

    def final_clusters(self):
        l = []

        for clusters in self.clusters:
            appending = []
            for point_index in clusters:
                appending.append(self.points_list[point_index])
            l.append(appending)

        return l

    def final_centroids(self):
        return self.centroids
