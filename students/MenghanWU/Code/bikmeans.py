# -*- coding:utf-8 -*-

import colour as colour
import numpy
import random
import matplotlib.pyplot as plt
import matplotlib
import math
import data_process
import copy

import data_process
import K_means


def get_sse(data, centroid, q):
    sse = []
    for i in range(len(centroid)):
        sum = 0
        for j in range(len(q)):
            if q[j] == i:
                sum += K_means.get_dist(data[j][0], data[j][1], centroid[i][0], centroid[i][1])
        sse.append(sum)
    return sse


def max_key(data):
    maxptr = 0
    for i in range(len(data)):
        if (data[i] > data[maxptr]):
            maxptr = i
    return maxptr


def bi_kmeans(data, k):
    q = K_means.kmeans(data, 2)
    index = 2
    # 初始化centroids节点
    centroid = K_means.init_centroids(data, 2)
    index=2
    while index!=k:
        datasep = []
        a = max_key(get_sse(data, centroid, q))
        print a
        for i in range(len(data)):
            if q[i] == a:
                datasep.append(data[i])
        sep_centroid = K_means.init_centroids(datasep, 2)
        index += 1
        print centroid
        centroid=copy.deepcopy(numpy.delete(centroid, a,0))
        print centroid
        centroid = numpy.concatenate((centroid, sep_centroid))
        print "======================================"
        print centroid
        print "======================================"
        q = K_means.cost_funct(data, centroid)
        centroid = K_means.update_centroid(data, centroid, q)
    pre_centroid = centroid
    q = K_means.cost_funct(data_process.data, centroid)
    now_centroid = copy.deepcopy(pre_centroid)
    pre_centroid = K_means.update_centroid(data, pre_centroid, q)
    print now_centroid - pre_centroid
    while (now_centroid != pre_centroid).any():
        print pre_centroid
        print now_centroid
        q = K_means.cost_funct(data, pre_centroid)
        print q
        now_centroid = copy.deepcopy(pre_centroid)
        pre_centroid = K_means.update_centroid(data, pre_centroid, q)
    return q


print bi_kmeans(data_process.data, 7)
data_process.draw(data_process.x, data_process.y, bi_kmeans(data_process.data,7))