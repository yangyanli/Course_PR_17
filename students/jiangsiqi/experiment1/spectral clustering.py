#coding = UFT-8  
import numpy as np
import pylab as pl
import math
import matplotlib.pyplot as plt 
import random

def getWbyKNN(data,k):
    l=len(data)
    distance=np.zeros((l,l))
    W =np.zeros((l,l))
    for i in range(0,l):
        for j in range(i+1,l):
            dis=np.linalg.norm((data[i])-(data[j]))
            distance[i][j]=dis
            distance[j][i]=dis
    for idx,each in enumerate(distance): 
            index_array  = np.argsort(each)
            W[idx][index_array[1:k+1]] = 1  # 距离最短的k个点
    tmp_W = np.transpose(W)
    W = (tmp_W+W)/2 
    return W
def getD(W):    # 获得度矩阵
    points_num = len(W)
    D = np.diag(np.zeros(points_num))
    for i in range(0,points_num):
        for j in range(0,points_num):
            D[i][i] = D[i][i]+W[i][j]
    return D
def getEigVec(L,cluster_num):  #从拉普拉斯矩阵获得特征矩阵
    eigval,eigvec = np.linalg.eig(L)#特征值与向量
    dim = len(eigval)
    dictEigval = dict(zip(eigval,range(0,dim)))
    kEig = np.sort(eigval)[0:cluster_num]
    ix = [dictEigval[k] for k in kEig]
    return eigval[ix],eigvec[:,ix]
def randRGB():
    return (random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0)

def getMeans(mask1,c,cluster_data):
    centerl=len(c)
    su=c
    for i in range(0,centerl):
        ml=len(mask1[i])
        if(ml>1):
            for j in range(1,ml):
                su[i]=su[i]+cluster_data[mask1[i][j]]
            su[i]=su[i]/(ml-1)
        else:
            su[i]=c[i]
    return su
def getVar(mask1,c,cluster_data):
    centerl=len(c)
    sum=0
    for i in range(0,centerl):
        ml=len(mask1[i])
        if(ml>1):
            for j in range(1,ml):
                s=np.linalg.norm(cluster_data[mask1[i][j]]-c[i])
                sum=s+sum
    return sum

def KMeans(k,cluster_data):
    cl=len(cluster_data)
    dl=len(cluster_data[0])
    mask1 = [[] for i in range(k)]
    c=np.zeros((k,dl))
    for i in range(0,k):
        c[i]=cluster_data[random.randint(0,cl)]#随机获得质心
        mask1[i].append(-1)

    for i in range(0,cl):#根据质心决定当前元素属于哪个簇
        di=0
        di=di+np.linalg.norm(cluster_data[i]-c[0])
        m=0
        for j in range(1,k):
            dc=0
            dc=dc+np.linalg.norm(cluster_data[i]-c[j])
            if(dc<di):
                di=dc
                m=j
        mask1[m].append(i)        

    oldVar=-1
    newVar=getVar(mask1,c,cluster_data)
    while(abs(newVar-oldVar)>=1):
        c=getMeans(mask1,c,cluster_data)
        oldVar=newVar
        newVar=getVar(mask1,c,cluster_data)
        mask1=[[] for i in range(k)]
        for i in range(0,k):
            mask1[i].append(-1)
        for i in range(0,cl):#根据质心决定当前元素属于哪个簇
            di=0
            di=di+np.linalg.norm(cluster_data[i]-c[0])
            m=0
            for j in range(1,k):
                dc=0
                dc=dc+np.linalg.norm(cluster_data[i]-c[j])
                if(dc<di):
                    di=dc
                    m=j
            mask1[m].append(i)   
    return mask1
x=[]
y=[]
data2=[]
fd= open('E:\\moshi\\experiment1\\data\\synthetic_data\\mix.txt','r')

for line in fd.readlines():
    a=list(map(float,line.split(',')))
    data2.append([a[0],a[1]])
   
data=np.array(data2)
cluster_num = 28
KNN_k = 5
W = getWbyKNN(data,KNN_k)
D = getD(W)
L = D-W
eigval,eigvec = getEigVec(L,cluster_num)

mask= KMeans(cluster_num,eigvec)
center=np.zeros((cluster_num,2))
for i in range(0,cluster_num):
    ml=len(mask[i])
    if(ml>1):
       for j in range(1,ml):
            center[i]=center[i]+data[mask[i][j]]
       center[i]=center[i]/(ml-1)     

print(center)
for i in range(0,len(center)):
    colors=randRGB()
    ml=len(mask[i])
    if(ml>1):
        for j in range(1,len(mask[i])):
            plt.plot(data[mask[i][j]][0],data[mask[i][j]][1],'o',color=colors)
        plt.plot(center[i][0],center[i][1],"rx") 

plt.show()
