#coding=utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
import random
import math
import sys
from sklearn.metrics import adjusted_mutual_info_score

##  load  mnist 
def load_mnist():
    filename = 'train-images.idx3-ubyte'
    binfile = open(filename , 'rb')
    buf = binfile.read()
     
    index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')

    numImages = 1000
    data = np.zeros([numImages,numColumns*numRows],dtype = np.int32)
    for i in range(numImages):
    	
    	im = struct.unpack_from('>784B' ,buf, index)
    	index += struct.calcsize('>784B')
    	im = np.array(im)

    	data[i,:]=im
    filename = 'train-labels.idx1-ubyte'
    binfile = open(filename , 'rb')
    buf = binfile.read()
     
    index = 0
    magic_label, numLabels  = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')

    numLabels = numImages
    label = np.zeros([numLabels],dtype = np.int32)

    for i in range(numLabels):
        
        im = struct.unpack_from('>B' ,buf, index)
        index += struct.calcsize('>B')
        im = np.array(im)

        label[i]=im
    return data,label

def load_data(path, feature_num=2):
    f = open(path)
    data = []
    label = []
    for line in f.readlines():
        lines = line.strip().split(",")
        tmp = []
        if len(lines) != feature_num+1:
            continue
        for i in xrange(feature_num):
            tmp.append(float(lines[i]))
        label.append(int(lines[feature_num]))
        data.append(tmp)
    f.close()
    data = np.array(data)
    label = np.array(label)
    return data,label


### image show

# im = im.reshape(28,28)
# fig = plt.figure()
# plotwindow = fig.add_subplot(111)
# plt.imshow(im , cmap='gray')
# plt.show()




def initCompetition(n , m , d):

    array = np.random.random(size=n * m *d)
    com_weight = array.reshape(n,m,d)
    return com_weight


def cal2NF(X):
    res = 0
    for x in X:
        res += x*x
    return res ** 0.5


def normalize(dataSet):
    for data in dataSet:
        two_NF = cal2NF(data)
        for i in range(len(data)):
            data[i] = data[i] / two_NF
    return dataSet 

def normalize_weight(com_weight):
    for x in com_weight:
        for data in x:
            two_NF = cal2NF(data)
            for i in range(len(data)):
                data[i] = data[i] / two_NF
    return com_weight


def getWinner(data , com_weight):
    max_sim = 0
    n,m,d = np.shape(com_weight)
    mark_n = 0
    mark_m = 0
    for i in range(n):
        for j in range(m):
            if sum(data * com_weight[i,j]) > max_sim:
                max_sim = sum(data * com_weight[i,j])
                mark_n = i
                mark_m = j
    return mark_n , mark_m


def getNeibor(n , m , N_neibor , com_weight):
    res = []
    nn,mm , _ = np.shape(com_weight)
    for i in range(nn):
        for j in range(mm):
            N = int(((i-n)**2+(j-m)**2)**0.5)
            if N<=N_neibor:
                res.append((i,j,N))
    return res


def eta(t,N):
    return (0.3/(t+1))* (math.e ** -N)


def do_som(dataSet , com_weight, T , N_neibor):
    for t in range(T-1):
        if(t%10==0):
            sys.stdout.write('iter : {}\r'.format(t))
        sys.stdout.flush()
        com_weight = normalize_weight(com_weight)
        for data in dataSet:
            n , m = getWinner(data , com_weight)
            neibor = getNeibor(n , m , N_neibor , com_weight)
            for x in neibor:
                j_n=x[0];j_m=x[1];N=x[2]

                com_weight[j_n][j_m] = com_weight[j_n][j_m] + eta(t,N)*(data - com_weight[j_n][j_m])
            N_neibor = N_neibor+1-(t+1)/200
    res = {}
    N , M , _ = np.shape(com_weight)
    for i in range(len(dataSet)):
        n, m = getWinner(dataSet[i], com_weight)
        key = n*M + m
        if res.has_key(key):
            res[key].append(i)
        else:
            res[key] = []
            res[key].append(i)
    return res
def reverse_res(dataset,res):
    
    length = len(dataset)
    label = np.zeros([length])
    for x in res:
        for y in res[x]:
            label[y]= x
    return label


def SOM(dataSet,com_n,com_m,T,N_neibor):
    dataSet = normalize(dataSet)
    com_weight = initCompetition(com_n,com_m,len(dataSet[0]))
    C_res = do_som(dataSet, com_weight, T, N_neibor)
    res = reverse_res(dataSet,C_res)
    return C_res,res


data, label = load_data('synthetic_data/flame.txt',2)
dataSet = data.copy()
_,res = SOM(dataSet,3,3,2000,2)

fig = plt.figure(figsize=(10, 15))
img0 = fig.add_subplot(421)
img0.set_title("orignal")
plt.scatter(data[:,0], data[:,1],c=label)
img1 = fig.add_subplot(422)
plt.scatter(data[:,0], data[:,1 ],c=res)
img1.set_title("SOM")


# data, label = load_data('synthetic_data/flame.txt',2)
# dataSet = data.copy()
# _,res = SOM(dataSet,1,2,200,2)

# img0 = fig.add_subplot(423)
# img0.set_title("orignal")
# plt.scatter(data[:,0], data[:,1],c=label)
# img1 = fig.add_subplot(424)
# plt.scatter(data[:,0], data[:,1 ],c=res)
# img1.set_title("SOM")




# data, label = load_data('synthetic_data/R15.txt',2)
# dataSet = data.copy()
# _,res = SOM(dataSet,3,5,20000,2)

# img0 = fig.add_subplot(425)
# img0.set_title("orignal")
# plt.scatter(data[:,0], data[:,1],c=label)
# img1 = fig.add_subplot(426)
# plt.scatter(data[:,0], data[:,1 ],c=res)
# img1.set_title("SOM")



# data, label = load_data('synthetic_data/mix.txt',2)
# dataSet = data.copy()
# _,res = SOM(dataSet,6,4,20,2)

# img0 = fig.add_subplot(427)
# img0.set_title("orignal")
# plt.scatter(data[:,0], data[:,1],c=label)
# img1 = fig.add_subplot(428)
# plt.scatter(data[:,0], data[:,1 ],c=res)
# img1.set_title("SOM")

# data,label = load_mnist()
# _,res = SOM(data,2,5,2000,2)


# acc = adjusted_mutual_info_score(label,res)
# print acc

plt.show()
