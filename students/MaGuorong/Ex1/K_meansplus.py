# -*- coding: utf-8 -*-

import numpy as np
import random
import math


def euclidean_distance(point1, point2):
    # 计算列表中任意两个点的欧氏距离
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)

    return np.linalg.norm(point2 - point1)  # 计算欧式距离


def weighted_values(list):
    sum = np.sum(list)
    return [x / sum for x in list]


def choose_weighted(distance_list):
    distance_list = [x ** 2 for x in distance_list]
    weighted_list = weighted_values(distance_list)
    indices = [i for i in range(len(distance_list))]
    return np.random.choice(indices, p=weighted_list)


class K_means_plus_plus:
    ##初始化 类似于Java的构造方法 points_list是数据集  ,k是聚类的数量
    def __init__(self, points_list, k):
        self.centroid_count = 0
        self.point_count = len(points_list)
        self.cluster_count = k
        self.points_list = list(points_list)  # 将元组转化为列表
        self.initialize_random_centroid()
        self.initialize_other_centroid()


    def initialize_random_centroid(self):
        # 生成第一个种子点  从数据库中随机挑一个点
        print ("ddd")
        self.centroid_list = []
        index = random.randint(0, len(self.points_list) - 1)

        self.centroid_list.append(self.remove_point(index))
        self.centroid_count = 1

    def initialize_other_centroid(self):

        # 生成其他的种子点，先取一个能落在Sum(D(x))中的随机值random,然后用random-=d(x)
        # ,直到其小于等于0,此时的点就是下一个种子点
        print ("ddd")
        while not self.is_finished():
            distance = self.find_smallest_distance()
            chosen_index = choose_weighted(distance)
            self.centroid_list.append(self.remove_point(chosen_index))
            self.centroid_count += 1

    def remove_point(self, index):
        new_centroid = self.points_list[index]
        del self.points_list[index]

        return new_centroid

    def is_finished(self):
        outcome = False
        if self.centroid_count == self.cluster_count:
            outcome = True
        return outcome

    def find_smallest_distance(self):
        distance_list = []

        for point in self.points_list:
            distance_list.append(self.find_nearest_centroid(point))

        return distance_list

    def find_nearest_centroid(self, point):
        min_distance = np.inf

        for values in self.centroid_list:
            distance = euclidean_distance(values, point)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def final_centroids(self):
        return self.centroid_list
