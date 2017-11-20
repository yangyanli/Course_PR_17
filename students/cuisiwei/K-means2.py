#coding=utf-8
#!/usr/bin/env python

from random import shuffle
from numpy import *
import matplotlib.pyplot as plt

def calcdist(x1,y1,x2,y2):
    return sqrt((x1-x2)**2+(y1-y2)**2)



def generateCent(data,k):
    minX = min(data[:,0])
    maxX = max(data[:, 0])
    minY = min(data[:,1])
    maxY = max(data[:, 1])
    cent = mat(zeros((k, 2)))
    #print cent
    for i in range(k):
        cent[i,:] = data[random.randint(0,data.shape[0]),:]

    return cent



if __name__=="__main__":
    filen = 'Aggregation'
    filename = r'D:\\synthetic_data\\'+filen+'.txt'
    k = 7

    file = loadtxt(filename,delimiter=',')
    data = file[:,0:2]
    data = array(data)
    cnt = data.shape[0] #num of points
    cluster = array(zeros((cnt, 1)))
    for i in range(cnt):
        cluster[i]=-1
    dist = mat(zeros((1,2)))
    #print "dist=",dist
    centroids = generateCent(data,k)
    Changed = True
    while Changed:
        #print "****************************"
        Changed = False
        for i in range(cnt):
            #print i
            dist = [1000000000000,-1]
            for j in range(k):
                tmp = calcdist(data[i,0],data[i,1],centroids[j,0],centroids[j,1])
                if tmp<dist[0]:
                    dist[0]=tmp
                    dist[1]=j
            #print cluster[i]
            if dist[1]!=cluster[i]:
                cluster[i]=dist[1]
                #print dist[1]
                Changed = True

        for i in range(k):
            sumofpoints = 0
            sumX = 0
            sumY = 0
            for j in range(cnt):
                if cluster[j]==i:
                    ++sumofpoints
                    sumX+=data[j,0]
                    sumY+=data[j,1]
            if sumofpoints!=0:
                centroids[i,0]=sumX/float(sumofpoints)
                centroids[i, 1] = sumY / float(sumofpoints)

    plt.ylabel("y")
    plt.xlabel("x")
    plt.rc('font', size=8)
    #print cluster
    #print data[:,0]
    # for i in range(cnt):
    #     print int(cluster[i])
    #     plt.scatter(data[i,0], data[i,1])
    plt.scatter(data[:,0], data[:,1],c=cluster[:,0])
    # tt= range(k)
    # X  =array(centroids[:, 0])
    # Y = array(centroids[:, 1])
    #plt.scatter(X[:,0],Y[:,0] , c=tt,marker='v',s=20)

    plt.savefig("k-means2 "+filen+".png")
    plt.show()


