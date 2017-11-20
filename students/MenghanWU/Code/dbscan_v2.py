# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import data_process

UNCLASSIFIED = False
NOISE = 0


def loadDataSet(fileName, splitChar='\t'):
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet


def get_vector_dist(point1, point2):
    return math.sqrt(np.power(point1 - point2, 2).sum())


def is_neigbor(point1, point2, domain):
    return get_vector_dist(point1, point2) <= domain


def domain_list(data, i, domain):
    radius_point = []
    for j in range(len(data)):
        if is_neigbor(data[:, i], data[:, j], domain):
            radius_point.append(i)
    return radius_point


def divided_cluster(data, now_cluster, wait_point, clusterID, density, domain):
    area = domain_list(data, wait_point, domain)
    if len(area) < density:  # 不满足minPts条件的为噪声点
        now_cluster[wait_point] = NOISE
        return False
    else:
        now_cluster[wait_point] = clusterID  # 划分到该簇
        for seedId in area:
            now_cluster[seedId] = clusterID

        while len(area) > 0:  # 持续扩张
            currentPoint = area[0]
            radius_point = domain_list(data, currentPoint, domain)
            if len(radius_point) >= density:
                for i in range(len(radius_point)):
                    resultPoint = radius_point[i]
                    if now_cluster[resultPoint] == UNCLASSIFIED:
                        area.append(resultPoint)
                        now_cluster[resultPoint] = clusterID
                    elif now_cluster[resultPoint] == NOISE:
                        now_cluster[resultPoint] = clusterID
            area = area[1:]
        return True


def dbscan(data, domain, radius):
    clusterId = 1
    num = len(data)
    now_cluster = [UNCLASSIFIED] * num
    for pointId in range(num):
        print now_cluster
        point = data[:, pointId]
        if now_cluster[pointId] == UNCLASSIFIED:
            if divided_cluster(data, now_cluster, pointId, clusterId, domain, radius):
                clusterId = clusterId + 1
    return now_cluster, clusterId - 1


def plotFeature(data, clusters, clusterNum):
    nPoints = data.shape[1]
    matClusters = np.mat(clusters).transpose()
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(clusterNum + 1):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
        ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=50)


def main():
    dataSet = loadDataSet('Aggregation.txt', splitChar=',')
    dataSet = np.mat(dataSet).transpose()
    print(dataSet)
    print dataSet.shape[1]
    clusters, clusterNum = dbscan(dataSet, 4, 10)
    print("cluster Numbers = ", clusterNum)
    # print(clusters)
    plotFeature(dataSet, clusters, clusterNum)


main()
plt.show()
