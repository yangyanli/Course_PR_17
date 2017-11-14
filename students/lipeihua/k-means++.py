#!/usr/bin/env python
# -*- coding:utf-8 -*-  
from math import pi, sin, cos
from collections import namedtuple
from random import random, choice
from copy import copy 
import matplotlib.pyplot as plt 

try:
    import psyco
    psyco.full()
except ImportError:
    pass
 
 
FLOAT_MAX = 1e100
 
 
class Point:
    __slots__ = ["x", "y", "group"]#一个点有x，y，所属的组号
    def __init__(self, x=0.0, y=0.0, group=0):#一开始，每个点的坐标都初始化为（0,0），组号为0
        self.x, self.y, self.group = x, y, group
 
 
def generate_points(dataSet,npoints):
    points = [Point() for _ in xrange(npoints)]
    i=0;
    for p in points:            #读取数据集中点的坐标
        p.x = dataSet[i][0]
        p.y = dataSet[i][1]
        i=i+1
    return points
 
 
def nearest_cluster_center(point, cluster_centers):
    def sqr_distance_2D(a1, a2):#求两点之间的距离
        return (a1.x - a2.x) ** 2  +  (a1.y - a2.y) ** 2
 
    min_index = point.group
    min_dist = FLOAT_MAX
 
    for i, cc in enumerate(cluster_centers):#每个点到聚类中心的最短距离
        d = sqr_distance_2D(cc, point)
        if min_dist > d:
            min_dist = d
            min_index = i
 
    return (min_index, min_dist)
 
 
def kpp(points, cluster_centers):#初始化聚类中心，k-means是随机的，k-means++尽可能取相距较远的点
    cluster_centers[0] = copy(choice(points))
    d = [0.0 for _ in xrange(len(points))]
 
    for i in xrange(1, len(cluster_centers)):
        sum = 0
        for j, p in enumerate(points):
            d[j] = nearest_cluster_center(p, cluster_centers[:i])[1]
            sum =sum+ d[j]
 
        sum=sum* random()#某个点距离当前聚类中心的距离越远，它被选中的可能性更大
 
        for j, di in enumerate(d):
            sum =sum- di
            if sum > 0:
                continue
            cluster_centers[i] = copy(points[j])
            break
 
    for p in points:
        p.group = nearest_cluster_center(p, cluster_centers)[0]#第一次聚类
 
 
def lloyd(points, nclusters):#k-means
    cluster_centers = [Point() for _ in xrange(nclusters)]#声明聚类中心
 
    kpp(points, cluster_centers)#初始化聚类中心
 
    lenpts10 = len(points) >> 10#有90%以上的点没改变组号，就认为可以结束聚类了
 
    changed = 0
    while True:
        for cc in cluster_centers:
            cc.x = 0
            cc.y = 0
            cc.group = 0
 
        for p in points:
            cluster_centers[p.group].group += 1#求每个组中的点数
            cluster_centers[p.group].x += p.x#求组内各点到该组聚类中心在x方向上的距离
            cluster_centers[p.group].y += p.y#求组内各点到该组聚类中心在y方向上的距离
 
        for cc in cluster_centers:
            cc.x /= cc.group#求组内各点到该组聚类中心在x方向上的平均距离，作为该组新的聚类中心的x
            cc.y /= cc.group#求组内各点到该组聚类中心在y方向上的平均距离，作为该组新的聚类中心的y
 
        changed = 0
        for p in points:
            min_i = nearest_cluster_center(p, cluster_centers)[0]
            if min_i != p.group:
                changed += 1#计算改变组号的点的数目
                p.group = min_i
 
        if changed <= lenpts10:#改变组号的点的数目小于10%，就认为聚类结束
            break
 
    for i, cc in enumerate(cluster_centers):
        cc.group = i#之前聚类中心的group值用来记录组内点的数目，现在变成标记该聚类中心属于哪个组
 
    return cluster_centers
 
 
def print_eps(points, cluster_centers):
    mark = ['r','y','g','b','m','c','w','k']#只有8种颜色
    x=[]
    y=[]
    group=[]
    for p in points:
        x.append(p.x)
        y.append(p.y)
        group.append(mark[p.group%8])
    plt.scatter(x,y,c=group)#画点
    plt.show()

def main():
    npoints=0;#点数
    k = 6 # 聚类数
    dataSet = []  
    fileIn = open('R15.txt')  
    for line in fileIn.readlines():  
            npoints=npoints+1
            lineArr = line.strip().split(',')  
            dataSet.append([float(lineArr[0]), float(lineArr[1])])  
    points = generate_points(dataSet,npoints)#初始化聚类中心
    cluster_centers = lloyd(points, k)#k-means
    print_eps(points, cluster_centers)#画点
main()
