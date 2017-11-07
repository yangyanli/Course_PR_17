'''
It is implemented according to Spectral clustering algorithm!
'''

from Kmeansandkmeansplusp import ReadFile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def Gweight(data,sigma):
    n=len(data)
    W=np.zeros((n,n))
    for i in range(n):
        x1 = np.array(data[i])
        for j in range(i+1,n):
            x2=np.array(data[j])
            d=np.exp(-np.sum(np.power(x1-x2,2))/(2*sigma*sigma))
            W[i][j]=W[j][i]=d

    D=np.sum(W,axis=1)
    D=np.diag(D)
    L=D-W
    L=np.dot(np.dot(np.sqrt(D),W),np.sqrt(D))
    return W,D,L

def GetEig(L,dimension):
     Evalue, Evector = np.linalg.eig(L)
     index=np.argsort(Evalue)

     Evec=Evector[:,index[0:dimension]]

     return Evec

'''
data=ReadFile("DATA/flame.txt")

W,D,L=Gweight(data,1)
Evec=GetEig(L,50)
print(type(Evec))
print(Evec.shape)
c=KMeans(n_clusters=2).fit(Evec)

color = ['black', 'gray', 'red', 'darkgreen', 'blue', 'm', 'crimson', 'lime', 'coral', 'snow', 'yellow', 'teal',
         'lightpink', 'orange',
         'peru', 'blueviolet', 'skyblue']
fig, ax = plt.subplots()

for i in range(len(data)):

    ax.scatter(data[i][0], data[i][1], c=color[c.labels_[i]],
               alpha=0.3)


ax.grid(True)
plt.show()

'''