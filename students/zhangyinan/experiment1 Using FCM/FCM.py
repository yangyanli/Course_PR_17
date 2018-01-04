# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 09:36:27 2018

@author: Nancy
"""
import matplotlib.pyplot as plt
import math
import time
import random

def loadDataSet(fileName, splitChar='\t'):
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            curline.pop()
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet

def FCM(dataSet,cnumber):
    maxi=50#迭代次数
    #初始化U
    U=[]
    for i in range(len(dataSet)):
        cur =[]
        maxb=1
        total=0
        for j in range(cnumber):
            if(j!=cnumber-1):
                val=random.uniform(0,maxb)
                total+=val
                maxb=1-total
            else:
                val=1-total
            cur.append(val)
        U.append(cur)
    
    for i in range(maxi):
        #更新C
        C=[]
        for j in range(cnumber):
            cur=[]
            for ii in range(len(dataSet[0])):
                n=0.0
                d=0.0
                for k in range(len(dataSet)):
                    n+=(U[k][j]**2)*dataSet[k][ii]
                    d+=(U[k][j]**2)
                cur.append(n/d)
            C.append(cur)
        #更新j
        dis=[]
        for ii in range(len(dataSet)):
            cur=[]
            for j in range(cnumber):
                cur.append(distance(dataSet[ii],C[j]))
            dis.append(cur)
        #更新U
        for j in range(cnumber):
            for ii in range(len(dataSet)):
                d=0.0
                for k in range(cnumber):
                    d+=(dis[ii][j]/dis[ii][k])**2
                U[ii][j]=1/d
    cluster=[]
    for i in range(0,len(U)):
        maximum = max(U[i])
        for j in range(0,len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
                cluster.append(j)
    return U,cluster

def distance(p,c):
    if len(p) !=len(c):
        return -1
    d=0.0
    for i in range(len(p)):
        d += abs(p[i]-c[i])**2
    return math.sqrt(d)
def plotFeature(datalist, clusters, clusterNum):
    nPoints = len(datalist)
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'brown', 'purple', 'orange','red']
    ax = fig.add_subplot(111)
    for co in range(0,8):
        x=[]
        y=[]
        for i in range(nPoints):
            if ((clusters[i]) % len(scatterColors)) == co:
                x.append(datalist[i][0])
                y.append(datalist[i][1])
        ax.scatter(x,y,c=scatterColors[co],marker = 'o',s=50)

def main():
    dataSet = loadDataSet('C:/Users/Nancy/Desktop/模式识别/Course_PR_17/experiment1/data/synthetic_data/mix.txt', splitChar=',')
    clusterNum=17
    U,cluster=FCM(dataSet,clusterNum)
    plotFeature(dataSet, cluster, clusterNum)
    


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))
