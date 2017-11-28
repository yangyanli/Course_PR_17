
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-


import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math
import time

UNCLASSIFIED = -1
NOISE = 0

# 读取数据，直接用写好方法
def loadData(fileName):
    dataset = []
    with open(fileName) as fr:
        for line in fr.readlines():
            cur = line.strip().split(",")
            flt = list(map(float, cur))
            dataset.append(flt)
    return dataset

# 欧氏距离
def getDistance(va, vb):
    return sqrt(sum(power(va - vb, 2)))

#判断是否在邻域内
def isNeighbor(va, vb, eps):
    return getDistance(va, vb) < eps

def getNeighbor(dataset, point, eps):
    #所有点的个数
    m = dataset.shape[1]
    #这个点的邻居集合
    neighbors = []
    
    #遍历所有点判断是否是邻居,存储的是这个节点的index
    for i in range(m):
        if isNeighbor(dataset[:, point], dataset[:, i], eps):
            neighbors.append(i)
    return neighbors

def getCluster(dataset, clusterResult, point, clusterId, eps, minPts):
# 分类，提取出噪音
    neighbors = getNeighbor(dataset, point, eps)
    if len(neighbors) < minPts: 
#     邻居数量小于minpts，属于噪音，不分类
        clusterResult[point] = NOISE
        return False
    else:
#         先把这个点和它的所有邻居都划分到这个簇
        clusterResult[point] = clusterId 
        for n in neighbors:
            clusterResult[n] = clusterId

        while len(neighbors) > 0: 
#             每次都是把第一个节点去掉了，所以每次都取第一个
            cur = neighbors[0]
#           对邻居中的每一个点，求它们的邻居集合nei2
            nei2 = getNeighbor(dataset, cur, eps)
#           如果这个邻居是核心对象
            if len(nei2) >= minPts:
#              遍历这个邻居的邻居
                for i in range(len(nei2)):
                    curNei = nei2[i]
#             判断是否未分类，防止有两个点的邻居是重复的，加到neighbors集的末尾，相当于一个队列
                    if clusterResult[curNei] == UNCLASSIFIED:
                        neighbors.append(curNei)
#             这个点被加入了point的邻居，那么把这个点所属的类改成clusterID
                        clusterResult[curNei] = clusterId
#             如果之前被判断为噪音了，那么现在给它找到了邻居，也加入
                    elif clusterResult[curNei] == NOISE:
                        clusterResult[curNei] = clusterId
#           无论如何，把这个处理过的节点去掉
            neighbors = neighbors[1:]
        return True

def dbscan(dataset, eps, minPts):
    clusterId = 1
#     所有数据个数
    m = dataset.shape[1]
#     初始化为false，用这个list来表明该点是否被归入某类
    clusterResult = [UNCLASSIFIED] * m
#     print(clusterResult)
    for i in range(m):
#         第i列，表示第i个点
        point = dataset[:, i]
#     如果未分类
        if clusterResult[i] == UNCLASSIFIED:
#         进行分类，返回值为true说明不是噪音，且找到了这一整类，类别数加一
            if getCluster(dataset, clusterResult, i, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1

# def plotFeature(data, clusters, clusterNum):
#     matClusters = mat(clusters).transpose()
#     fig = plt.figure()
#     scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange','brown']
#     ax = fig.add_subplot(111)
#     for i in range(clusterNum + 1):
#         colorStyle = scatterColors[i % len(scatterColors)]
#         subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
#         ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorStyle, s=50)

def show(datamat, clusters):
    from matplotlib import pyplot as plt
#     datamat = mat(datamat).transpose()
    num = datamat.shape[0]
#     print(num)
#     clusters = mat(clusters)
    markp = ['or', 'ob', 'og', 'om', 'oy', '+r', 'oc', '<r', 'sr', 'pr']  
    for i in range(num):
        marki = int(clusters[i])#marki是每一个点的质心序号（0，1，……，k-1）
        plt.plot(datamat[i,0], datamat[i,1], markp[marki])
#     markc = ['Dk', 'Dk', 'Dk', 'Dk', 'Dk', '+b', 'Dk', 'Dk', '<b', 'pb']
    plt.show()
        
        
def main():
    eps = 1
    minPts = 6
#     dataSet = loadData('agg_co.txt')
    dataSet = loadData('flame_co.txt')
    dataMat = mat(dataSet).transpose()
#     clusters, clusterNum = dbscan(dataMat, eps, minPts)  #agg 2 15
    clusters, clusterNum = dbscan(dataMat, eps, minPts)  #flame 1 6
    print("聚类数量：", clusterNum)
    print("eps:",eps)
    print("minPts:", minPts)
    dataMat2 = mat(dataSet)
    show(dataMat2, clusters)

if __name__ == '__main__':
    main()

