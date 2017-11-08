# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:36:46 2017

@author: zhuofeng
"""
import numpy as np
import random
import matplotlib.pyplot as plt

#欧式距离
def distEclud(vecA,vecB):
    return np.linalg.norm(vecA-vecB)

#确定k个最初的中心点
def randCenter(dataSet,k):
    n=np.shape(dataSet)[1]#n为点的列数
    centers=np.mat(np.zeros((k,n)))
    
    for col in range(n):
        mincol=min(dataSet[:,col])
        maxcol=max(dataSet[:,col])
        #产生k行1列的随机数：rand(k,1)
        centers[:,col]=np.mat(mincol+float(maxcol-mincol)*np.random.rand(k,1))
        
    return centers

#聚类主函数
def main():
    #从文件中读取数据 
    data=np.loadtxt('mix.txt',delimiter=',')
    position = data[:,0:2]
    dataSet=np.mat(position)
    m=np.shape(dataSet)[0]
    k=23
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
            
    #画结果图
    colors=[]
    for i in range(23):
        color = [random.random() for _ in range(3)]
        colors.append(color)
    for j in range(m):
        a = int(clustDist[j,0])
        color = colors[a]
        plt.scatter(dataSet[j,0],dataSet[j,1],c=color)
            
    plt.show()    
if __name__ == '__main__':
    main()