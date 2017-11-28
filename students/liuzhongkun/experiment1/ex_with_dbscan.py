import numpy as np
import matplotlib.pyplot as plt
import random as ran
import math

ep = 1.5
M = 8

def length(a,b):
    return math.sqrt(a*a + b*b)

def dbscan(data):
    cur_k = 1
    label = [0] * len(data)
    I = [i for i in range(len(data))]
    point = [[] for i in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data)):
            if length(data[i,0] - data[j,0],data[i,1] - data[j,1]) < ep:
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
    
data_raw = np.loadtxt('.\\mix.txt', \
                  delimiter = ',')
data = data_raw[:,0:2]
label = dbscan(data)

k = 0
for i in label:
    if i > k:
        k = i
k = k + 1
print(k)
'''
center = [[0,0] for i in range(k)]
point_num=[0]*k
for i in range(len(label)):
    if label[i] == -1:
        continue
    center[label[i]] = center[label[i]] + data[i]
    point_num[label[i]] = point_num[label[i]] + 1
for i in range(1,len(center)):
    center[i] = center[i]/float(point_num[i])
for i in range(len(label)):
    if label[i] == -1:
        min_l = 10000#int('Inf')
        min_c = -1
        for c in range(1,len(center)):
            tmp = (center[c][0] - data[i][0])*(center[c][0] - data[i][0]) \
                  + (center[c][1] - data[i][1])*(center[c][1] - data[i][1])
            tmp = tmp/math.sqrt(point_num[c])
            if tmp < min_l:
                min_l = tmp
                min_c = c
        label[i] = min_c
'''
col = ['ro','go','bo','co','ko','mo','yo']*10;
plt.figure(1)
for i in range(len(label)):
    if label[i] != -1:
        plt.plot(data[i,0],data[i,1],col[label[i]])
    else:
        plt.plot(data[i,0],data[i,1],'y+')
#plt.figure(2)
#for i in range(len(data)):
#    plt.plot(data[i,0],data[i,1],col[int(data[i,2] - 1)])

#plt.scatter(data[:,0],data[:,1],'ro')
plt.show()
