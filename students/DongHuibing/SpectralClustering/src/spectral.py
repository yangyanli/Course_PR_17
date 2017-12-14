#由于我对Python不熟悉，这份代码参考了陈安东的实现，然后我查阅了相关语法和库函数的用法，也做了一些具体实现上的改变
import numpy as np
from math import exp
from numpy.linalg import norm
from numpy.linalg import eig
#from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sigma = 0.5
k1 = 10
k2 = 20#number of clusters
#fo = open('..\\..\\data\synthetic_data\\Aggregation.txt')
#fo = open('..\\..\\data\synthetic_data\\flame.txt')
fo = open('..\\..\\..\\data\synthetic_data\\mix.txt')
#fo = open('..\\..\\data\synthetic_data\\R15.txt')

"""
#get the hyper-parameters
sigma = float(input(("Please enter sigma:")))
k1, k2 = map(int, input("Please enter k1 and k2:").split())

#read data set
fileName = input("Please enter the file name:")
fo = open('..\\..\\data\synthetic_data\\' + fileName)
"""

rawData = fo.readlines()
n = len(rawData)

#get the original points
pnts = np.zeros((n, 4))
for i in range(len(rawData)):
    data = rawData[i].split(",")
    pnts[i] = np.array([float(data[0]), float(data[1]), float(data[2]), 0])

#calculate similarity matrix
def getSimilarity(x, y):
    return exp(-norm(x - y) * 0.5 / sigma / sigma)
W = np.zeros((n, n))
for i in range(0, n):
    for j in range(i, n):
        W[i, j] = W[j, i] = getSimilarity(pnts[i, 0:2], pnts[j, 0:2])

#calculate degree matrix
D = np.zeros((n, n))
for i in range(0,n):
    D[i, i] = np.sum(W[i])

#calculate Laplacians matrix
L = D - W

#get the k1 max eigenvalues and eigenvector
eigenvalue, eigenvector = eig(L)
k1_max_id = np.argsort(eigenvalue)[0:k1]#'-' sign for descending sort
k1_max_eigenvalue = eigenvalue[k1_max_id]
k1_max_eigenvector = eigenvector[:,k1_max_id]

#k means
#whiten_eigenvector = whiten(k1_max_eigenvector)
#print(whiten_eigenvector)
#centroid = kmeans(whiten_eigenvector, k2)[0]
#label = vq(whiten_eigenvector, centroid)[0]
#pnts[:][3] = label

#k means
kmeans = KMeans(n_clusters = k2)
kmeans.fit(k1_max_eigenvector)
pnts[:, 3] = kmeans.labels_

#visualation
plt.scatter(pnts[:, 0], pnts[:, 1], c = pnts[:, 3])
plt.show()