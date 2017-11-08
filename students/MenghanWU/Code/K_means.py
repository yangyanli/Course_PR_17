# -*- coding:utf-8 -*-

import colour as colour
import numpy
import random
import matplotlib.pyplot as plt
import matplotlib
import math
import data_process
import copy


def init_centroids(data, k):
    init_centroid = []
    for i in range(k):
        index = random.randint(0, len(data) - 1)
        init_centroid.append(data[index])
    init_centroid = numpy.array(init_centroid)
    return init_centroid


init_centroid = init_centroids(data_process.data, 5)
pre_centroid = init_centroid


def get_dist(x1, y1, x2, y2):
    dist = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    return dist


def min_key(data):
    maxptr = 0
    for i in range(len(data)):
        if (data[i] < data[maxptr]):
            maxptr = i
    return maxptr


# 给点加标签
def cost_funct(data, centroid):
    q = []
    for j in range(len(data)):
        cost = []
        for s in range(len(centroid)):
            # print init_centroid[i][0], init_centroid[i][1], data[j][0], data[j][1]
            a = get_dist(centroid[s][0], centroid[s][1], data[j][0], data[j][1])
            cost.append(a)
        pt = min_key(cost)
        q.append(pt)
    return q


# 更新中心点
def update_centroid(data, centroid, q):
    for i in range(len(centroid)):
        sum_x = 0
        sum_y = 0
        count = 0
        for j in range(len(q)):
            if q[j] == i:
                count += 1
                sum_x += data[j][0]
                sum_y += data[j][1]
                centroid[i][0] = round(sum_x / count, 2)
                centroid[i][1] = round(sum_y / count, 2)
    return centroid


# print update_centroid(data_process.data, init_centroid, q)


def kmeans(data, k):
    init_cen = init_centroids(data, k)
    pre_cent = init_cen
    q = cost_funct(data_process.data, init_cen)
    now_centroid = copy.deepcopy(pre_cent)
    pre_cent = update_centroid(data, pre_cent, q)
    print "======================================"
    print now_centroid - pre_cent

    while (now_centroid - pre_cent != 0).any():
        print "======================================"
        q = cost_funct(data, pre_cent)
        print q
        now_centroid = copy.deepcopy(pre_cent)
        print now_centroid
        pre_cent = update_centroid(data, pre_cent, q)
        print (now_centroid -pre_cent)

    return q


#print kmeans(data_process.data, 5)
data_process.draw(data_process.x, data_process.y, kmeans(data_process.data, 2))
