import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
import numpy.linalg as ln
from math import sqrt,exp
from sklearn.cluster import KMeans

datamat = []
adjacent = []
sigma = 0.6
laplacians = []
k1=5
k2=2
def distance(u,v):
    return exp(0-(ln.norm(u-v,2)/(2*sigma*sigma)))


if __name__ == "__main__":
    sigma = float(input("sigma:"))
    k1 = int(input("k1:"))
    k2 = int(input("k2:"))
    file = open("D:\\PatternRecognition\\Course_PR_17\\experiment1\\data\\synthetic_data\\mix.txt") #read file
    datas = file.readlines()

    datamat = np.zeros((0,3))
    
    for data in datas:
        xyz = data.split(",")
        datamat = np.row_stack((datamat,[float(xyz[0]),float(xyz[1]),0]))

    adjacent = np.zeros((len(datas),len(datas)))

    for i in range(0,len(datas)):
        for j in range(i+1,len(datas)):
            adjacent[i,j]=adjacent[j,i]=distance(datamat[i,:],datamat[j,:])

    degree = np.zeros((len(datas),len(datas)))

    for i in range(0,len(datas)):
        degree[i,i] = np.sum(adjacent[i,:])

    laplacians = degree-adjacent

    eigvalue,eigvector = ln.eig(laplacians)

    k_indices = np.argsort(eigvalue)

    k_eigvalue = eigvalue[k_indices[0:k1]]
    k_eigvector = eigvector[:,k_indices[0:k1]]
    kms = KMeans(n_clusters=k2)
    kms.fit(k_eigvector)
    datamat[:,2] = kms.labels_
    print(kms.labels_)
    plt.scatter(datamat[:,0],datamat[:,1],c=datamat[:,2])
    plt.show()