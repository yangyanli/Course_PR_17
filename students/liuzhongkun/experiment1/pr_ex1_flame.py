import numpy as np
import matplotlib.pyplot as plt
import random as ran

ep = 1.2
M = 8

def dbscan(data):
    cur_k = 1
    label = [0] * len(data)
    I = [i for i in range(len(data))]
    point = [[] for i in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data)):
            if abs(data[j,0] - data[i,0]) < ep and abs(data[j,1] - data[i,1]) < ep:
                point[i].append(j)
    while len(I) > 0:
        tmp =ran.randint(0,len(I) - 1)
        cur_p = I[tmp]
        I = I[0:tmp] + I[(tmp+1):len(I)]
        if(label[cur_p] == 0):
            if(len(point[cur_p]) < M):
                label[cur_p] = -1
            else:
                T = point[cur_p]
                label[cur_p] = cur_k
                while len(T) > 0:
                    tmp = T[0]
                    T = T[1:len(T)]
                    if label[tmp] == 0 or label[tmp] == -1:
                        label[tmp] = cur_k
                        if len(point[tmp]) > M:
                            T = T + point[tmp]
                cur_k += 1
    return label       
    
data = np.loadtxt('.\\flame.txt', \
                  delimiter = ',')
label = dbscan(data)
col = ['ro','go','bo','co','ko','mo','yo']*10;
plt.figure(1)
for i in range(len(label)):
    if(label[i] != -1):
        plt.plot(data[i,0],data[i,1],col[label[i]])
    else:
        plt.plot(data[i,0],data[i,1],'y+')
#plt.figure(2)
#for i in range(len(data)):
#    plt.plot(data[i,0],data[i,1],col[int(data[i,2] - 1)])

#plt.scatter(data[:,0],data[:,1],'ro')
plt.show()
