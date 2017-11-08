from numpy import *
import matplotlib.pyplot as plt
import random
import math
import time

def loadDataSet(fileName):
	dataSet=[]
	fr=open(fileName)
	for line in fr.readlines():
		cur=line.strip().split(',')
		#flt=map(float,cur)
		#dataSet.append(flt)
		dataSet.append([float(cur[0]),float(cur[1])]) 
	return dataSet

def dist(vector1,vector2):
	return sqrt(sum(power(vector2-vector1,2)))

def randCenter(dataSet,k):
	n,dim=dataSet.shape
	cent=zeros((k,dim))
	for i in range(k):
		index=int(random.uniform(0,n))
		cent[i,:]=dataSet[index,:]
	return cent

def kMeans(dataSet,k):
	m=dataSet.shape[0]
	clusterAssment=mat(zeros((m,2)))
	cent=randCenter(dataSet,k)
	clusterChanged=True
	while clusterChanged:
		clusterChanged=False
		for i in range(m):
			minDist=inf
			minIndex=-1
			for j in range(k):
				distij=dist(cent[j,:],dataSet[i,:])
				if distij<minDist:
					minDist=distij
					minIndex=j
			if clusterAssment[i,0]!=minIndex:
				clusterChanged=True
				clusterAssment[i,:]=minIndex,minDist**2
		for i in range(k):
			points=dataSet[nonzero(clusterAssment[:,0].A==i)[0]]
			cent[i,:]=mean(points,axis=0)
	return cent,clusterAssment

def show(dataSet,k,cent,clusterAssment):
	colors=[]
	for i in range(100):
		color=[random.random() for j in range(3)]
		colors.append(color)
	numSamples,dim=dataSet.shape
	for i in range(numSamples):
		markIndex=int(clusterAssment[i,0])
		plt.scatter(dataSet[i,0],dataSet[i,1],c=colors[markIndex],marker='.')
	for i in range(k):
		plt.scatter(cent[i,0],cent[i,1],c=colors[i],marker='+',s=100)

def main():
	fileName=input("Input file name:\n")
	dataSet=loadDataSet(fileName)
	#dataSet=loadDataSet("R15.txt")
	dataSet=mat(dataSet)
	k=int(input("Input k:\n"))
	cent,clusterAssment=kMeans(dataSet,k)
	print (cent)
	show(dataSet,k,cent,clusterAssment)

if __name__=="__main__":
	start=time.clock()
	main()
	end=time.clock()
	print("finish all in %ss" % str(end-start))
	plt.show()
