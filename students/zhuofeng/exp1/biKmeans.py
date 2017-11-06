# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:51:41 2017
二分kmeans
@author: zhuofeng
"""

import numpy as np
import random
import matplotlib.pyplot as plt

def randCenter(dataSet,k):
    n=np.shape(dataSet)[1]#n为点的列数
    centers=np.mat(np.zeros((k,n)))
    
    for col in range(n):
        mincol=min(dataSet[:,col])
        maxcol=max(dataSet[:,col])
        #产生k行1列的随机数：rand(k,1)
        centers[:,col]=np.mat(mincol+float(maxcol-mincol)*np.random.rand(k,1))
        
    return centers

def kMeans(dataSet,k):
    m=np.shape(dataSet)[0]
    
    #一列存数据集对应的聚类中心号，一列存数据集到聚类中心的距离
    clustDist = np.mat(np.zeros((m,2)))
    centers = randCenter(dataSet,k)
    flag = True  #初始化迭代所需的标志位
#    counter=[]  #初始化计数器
    
    while flag:
        
        flag = False#如果后面没有更新聚类中心的话flag的值就不会被修改
        
        #找到每个点属于的类
        for i in range(m):
            datalist =[distEclud(centers[j,:],dataSet[i,:]) for j in range(k)]
#            for j in range(k):
#                datalist[j]=distEclud(centers[j,:],dataSet[i,:])
            mindist = min(datalist)
            minIndex = datalist.index(mindist)
            
            #找到新的聚类中心
            if clustDist[i,0]!=minIndex:
                flag=True
            
            #更新聚类中心
            clustDist[i,:]=minIndex,mindist
            
        #更新聚类中心
        for cent in range(k):
            #筛选出第cent类的点
            ptsInClust = dataSet[np.nonzero(clustDist[:,0].A==cent)[0]]
            #计算各列的均值,找到新的聚类中心
            centers[cent,:] = np.mean(ptsInClust,axis=0)
            
    return centers,clustDist

#欧式距离
def distEclud(vecA,vecB):
    return np.linalg.norm(vecA-vecB)

def main():
    #从文件中读取数据并存入矩阵 
    data = np.loadtxt('mix.txt',delimiter=',')
    position = data[:,0:2]
    dataSet = np.mat(position)
    
    k=23
    m=np.shape(dataSet)[0]
    center0 = np.mean(dataSet[:,1:]) #第一个聚类中心
    centers = [center0]
    
    #初始化聚类距离表
    ClustDist = np.mat(np.zeros((m,2)))
    for j in range(m):
        ClustDist[j,1]= distEclud(center0,dataSet[j,:])**2
        
    while(len(centers)<k):
        lowestSSE=np.inf #最小误差平方和
        
        for i in range(len(centers)):
            ptsInCurrCluster = dataSet[np.nonzero(ClustDist[:,0].A==i)[0],:]
            #用标准的kmeans算法，将ptsInCurrCluster划分成两个聚类中心，以及对应的聚类距离表
            centerMat,splitClustAss = kMeans(ptsInCurrCluster,2)
            #假设划分这个类，计算SSE会不会变小
            sseSplit = sum(ClustDist[np.nonzero(ClustDist[:,0].A!=i)[0],1])
            sseNotSplit =sum(ClustDist[np.nonzero(ClustDist[:0].A!=i)[0],1])
            #小于SSE就更新sse
            if(sseSplit+sseNotSplit)<lowestSSE:
                bestCenterToSplit = i
                bestNewCents = centerMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit+sseNotSplit
        
        bestClustAss[np.nonzero(bestClustAss[:,0].A==1)[0],0]=len(centers)
        bestClustAss[np.nonzero(bestClustAss[:,0].A==0)[0],0]=bestCenterToSplit
        
        #重构聚类中心
        centers[bestCenterToSplit]=bestNewCents[1,:].tolist()[0]
        centers.append(bestNewCents[1,:].tolist()[0])
        ClustDist[np.nonzero(ClustDist[:,0].A==bestCenterToSplit)[0],:]=bestClustAss
        
        colors=[]
    for i in range(23):
        color = [random.random() for _ in range(3)]
        colors.append(color)
    for j in range(m):
        a = int(ClustDist[j,0])
        color = colors[a]
        plt.scatter(dataSet[j,0],dataSet[j,1],c=color)
            
    plt.show()

if __name__ == '__main__':
    main()
