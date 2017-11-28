import numpy as np
import matplotlib.pyplot as plt
import random as ran


def cluster(data,k):
    center = np.array([[ran.random()*30,ran.random()*30] for i in range(k)])
    step = 100;
    label = [0 for i in range(len(data))]
    label_last = label[:]
    label_last[1] = 1
    point_of_c = [len(data)/k]*k
    while step > 0 and label_last != label:
        label_last = label[:]
        for i in range(len(data)):
            min_c = -1
            min_l = float('inf')
            for j in range(k):
                tmp = (1 + sum((data[i,0:2]-center[j])* \
                               ((data[i,0:2]-center[j]).T))) \
                       /np.sqrt(point_of_c[j] + 1)
                if tmp < min_l:
                    min_l = tmp
                    min_c = j
            label[i] = min_c
        sum_c = np.array([[0,0,0] for i in range(k)])
        for i in range(len(data)):
            sum_c[label[i]][0:2] = sum_c[label[i]][0:2] + data[i,0:2]
            sum_c[label[i]][2] += 1
        point_of_c = sum_c[:,2]
        for j in range(k):
            center[j] = sum_c[j][0:2] /(sum_c[j][2] + 1)
        step -= 1
    return label
data = np.loadtxt('.//Aggregation.txt',delimiter = ',')
label = cluster(data,7)
col = ['ro','go','bo','co','ko','mo','yo'];
plt.figure(1)
for i in range(len(data)):
    plt.plot(data[i,0],data[i,1],col[label[i]])
#plt.figure(2)
#for i in range(len(data)):
#    plt.plot(data[i,0],data[i,1],col[int(data[i,2] - 1)])

#plt.scatter(data[:,0],data[:,1],'ro')
plt.show()
