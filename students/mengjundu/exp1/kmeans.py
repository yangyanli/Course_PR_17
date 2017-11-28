#coding=utf-8
from numpy import *
import numpy as np
import random
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    data = np.loadtxt(fileName, delimiter=',')
    position = data[:, 0:2]
    dataMat = np.mat(position)
    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centers = np.mat(np.zeros((k, n)))

    for col in range(n):
        mincol = min(dataSet[:, col])
        maxcol = max(dataSet[:, col])
        centers[:, col] = np.mat(mincol + float(maxcol - mincol) * np.random.rand(k, 1))

    return centers


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print
        centroids
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def show(dataSet, k, centroids, clusterAssment):
    from matplotlib import pyplot as plt
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()


def main():
    dataMat = mat(loadDataSet('F:\Course_PR_17-master\experiment1\data\synthetic_data\Aggregation.txt'))
    myCentroids, clustAssing = kMeans(dataMat, 7)
    print
    myCentroids
    show(dataMat, 7, myCentroids, clustAssing)


if __name__ == '__main__':
    main()