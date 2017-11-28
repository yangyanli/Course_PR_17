import numpy as np
import matplotlib.pyplot as plt
import random as rd
import numpy.linalg as ln
from math import sqrt

klasses = 24
maxx = 100
maxy = 100
datamat = np.zeros((0,3))

def randomCenters():
    c = np.zeros((0,2))
    for i in range(0,klasses):
        c = np.row_stack((c,[rd.random()*maxx,rd.random()*maxy])) #.append([rd.random()*maxx,rd.random()*maxy])
    return c
def distance(u,v):
    return sqrt((u[0]-v[0])*(u[0]-v[0])+(u[1]-v[1])*(u[1]-v[1]))

def findNearestCenter(centers):
    for i in range(0,len(datamat)):
        min_dis = 0x7FFFFFFF
        min_center = 0x7FFFFFFF
        for j in range(0,len(centers)):
            if distance(datamat[i,:],centers[j,:])<min_dis:
                min_dis = distance(datamat[i,:],centers[j,:])
                min_center = j
        datamat[i,2] = min_center
def centerMove(centers):
    moving = np.zeros((klasses,3))
    for i in range(0,len(datamat)):
        moving[int(datamat[i,2]),0]+=datamat[i,0]
        moving[int(datamat[i,2]),1]+=datamat[i,1]
        moving[int(datamat[i,2]),2]+=1
    for i in range(0,klasses):
        if moving[i,2]>0:
            centers[i,0] = moving[i,0]/moving[i,2]
            centers[i,1] = moving[i,1]/moving[i,2]
    
def kmeans():
    centers = randomCenters()
    for i in range(0,50):
        findNearestCenter(centers)
        centerMove(centers)
    plt.scatter(centers[:,0],centers[:,1],marker="^")

if __name__ == "__main__":
    file = open("D:\\PatternRecognition\\Course_PR_17\\experiment1\\data\\synthetic_data\\mix.txt") #read file
    datas = file.readlines()

    for data in datas:
        xyz = data.split(",")
        datamat = np.row_stack((datamat,[float(xyz[0]),float(xyz[1]),0]))
    maxx = np.max(datamat[:,0])
    maxy = np.max(datamat[:,1])

    kmeans()

    plt.scatter(datamat[:,0],datamat[:,1],c=datamat[:,2])
    plt.show()