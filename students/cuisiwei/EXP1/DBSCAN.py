#coding=utf-8
#!/usr/bin/env python

from random import shuffle
from numpy import *
import matplotlib.pyplot as plt
import math


UNCLASSIFIED = False
NOISE = 0

def calcdist(x1,y1,x2,y2):
    return sqrt((x1-x2)**2+(y1-y2)**2)

def neighbor(x,y,eps):
	return math.sqrt(power(x-y,2).sum()) < eps

def calcCircle(data, curptr, eps):
    cnt = data.shape[1]
    res = []
    for i in range(cnt):
        if neighbor(data[:,curptr], data[:,i], eps):
            res.append(i)
    return res


def clustering(data, clusterres, curptr, id, eps, minpoints):
    pointsInCircle = calcCircle(data, curptr, eps)
    if len(pointsInCircle) < minpoints:
        clusterres[curptr] = NOISE
        return False
    else:
        clusterres[curptr] = id
        for ptr in pointsInCircle:
            clusterres[ptr] = id

        while len(pointsInCircle) > 0:
            cur = pointsInCircle[0]
            results = calcCircle(data, cur, eps)
            if len(results) >= minpoints:
                for i in range(len(results)):
                    curcnt = results[i]
                    if clusterres[curcnt] == UNCLASSIFIED:
                        pointsInCircle.append(curcnt)
                        clusterres[curcnt]=id
                    elif clusterres[curcnt] == NOISE:
                        clusterres[curcnt] = id
            pointsInCircle = pointsInCircle[1:]
        return True
def dbscan(data, eps, minpoints):
    clusterid = 1
    cnt = data.shape[1]
    clusterres = [UNCLASSIFIED] * cnt
    for point_id in range(cnt):
        point = data[:,point_id]
        if clusterres[point_id] == UNCLASSIFIED:
            if clustering(data, clusterres, point_id, clusterid, eps, minpoints):
                clusterid = clusterid + 1
    return clusterres,clusterid



if __name__=="__main__":
    # filen = 'Aggregation'
    # filename = r'D:\\synthetic_data\\' + filen + '.txt'
    # eps = 2
    # minpoints = 17

    # filen = 'flame'
    # filename = r'D:\\synthetic_data\\' + filen + '.txt'
    # eps = 2
    # minpoints = 23

    # filen = 'R15'
    # filename = r'D:\\synthetic_data\\' + filen + '.txt'
    # eps = 0.5
    # minpoints = 14

    filen = 'flame'
    filename = r'D:\\synthetic_data\\' + filen + '.txt'
    eps = 2
    minpoints = 23

    file = loadtxt(filename, delimiter=',')
    data = file[:, 0:2]
    data = mat(data).transpose()
    cnt = data.shape[1]  # after transpose, num of points
    # print cnt
    clusters ,clusterscnt= dbscan(data, eps, minpoints)

    clusters = mat(clusters).transpose()
    plt.ylabel("y")
    plt.xlabel("x")
    plt.rc('font', size=8)

    colors = []
    for i in range(clusterscnt):
        color = [random.random() for _ in range(3)]
        colors.append(color)


    for i in range(clusterscnt):
        color = colors[i]
        findCluster = data[:, nonzero(clusters[:, 0].A == i)]
        # print findCluster
        # print "\n"
        plt.scatter(findCluster[0, :].flatten().A[0], findCluster[1, :].flatten().A[0], c=color)

    plt.savefig("dbscan-"+filen + ".png")
    plt.show()