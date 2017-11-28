from numpy import *  
import time  
import matplotlib.pyplot as plt  
import kmeans
  
## step 1: load data  
print "step 1: load data"  
dataSet = []  
#fileIn = open('/Users/apple/Downloads/PR/data/synthetic_data/R15.txt') 
#fileIn = open('/Users/apple/Downloads/PR/data/synthetic_data/flame.txt') 
#fileIn = open('/Users/apple/Downloads/PR/data/synthetic_data/mix.txt')
fileIn = open('/Users/apple/Downloads/PR/data/synthetic_data/Aggregation.txt')  
for line in fileIn.readlines():  
    lineArr = line.strip().split(',')  
    dataSet.append([float(lineArr[0]), float(lineArr[1])])  
  
## step 2: clustering...  
print "step 2: cluster"  
dataSet = mat(dataSet)  
k = 8  
centroids, clusterset = kmeans.kmeans(dataSet, k)  
  
## step 3: show the result  
print "step 3: show the result"  
kmeans.visualization(dataSet, k, centroids, clusterset)  
