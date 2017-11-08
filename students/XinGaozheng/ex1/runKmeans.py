
from numpy import *  
import time  
import matplotlib.pyplot as plt 
import KMeans

print ("读取数据" )
dataSet = []
fileIn = open("Aggregation1.txt")
for line in fileIn.readlines(): 
	temp=[]
	lineArr = line.strip().split('\t')  #line.strip()把末尾的'\n'去掉
	temp.append(float(lineArr[0]))
	temp.append(float(lineArr[1]))
	dataSet.append(temp)
fileIn.close()
print ("聚类")
dataSet = mat(dataSet)
k = 7
centroids, clusterAssment = KMeans.kmeans(dataSet, k)
print ("完成聚类"  )
KMeans.showCluster(dataSet, k, centroids, clusterAssment)