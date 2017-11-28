#coding = UFT-8  
import numpy as np
import pylab as pl
import random
import math
import matplotlib.pyplot as plt 
import random
import Queue
from sklearn.cluster import KMeans


def randRGB():
    return (random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0)

def getMeans(mask1,c):
    cl=len(c)
    su=c
    for i in range(0,cl):
        ml=len(mask1[i])
        if(ml>1):
            for j in range(1,ml):
                su[i][0]=su[i][0]+mask1[i][j][0]
                su[i][1]=su[i][1]+mask1[i][j][1]
            su[i][0]=su[i][0]/(ml)
            su[i][1]=su[i][1]/(ml)
    return su
def getVar(mask1,c):
    cl=len(c)
    sum=0
    for i in range(0,cl):
        ml=len(mask1[i])
        for j in range(0,ml):
            sum=sum+math.sqrt((mask1[i][j][0]-c[i][0])*(mask1[i][j][0]-c[i][0])+(mask1[i][j][1]-c[i][1])*(mask1[i][j][1]-c[i][1]))
    return sum

data=[]
x=[]
c=[]
fd= open('E:\\moshi\\experiment1\\data\\synthetic_data\\mix.txt','r')
k=15
max_x=0
max_y=0

for line in fd.readlines():
    a=list(map(float,line.split(',')))
    data.append([a[0],a[1]])
    if(a[0]>max_x):
        max_x=a[0]
    if(a[1]>max_y):
        max_y=a[1]

l=len(data)
mask1 = [[] for i in range(k)]

for i in range(0,k):
    c.append([random.uniform(0, max_x),random.uniform(0,max_y)])#随机获得质心
    mask1[i].append([0,0])

for i in range(0,l):#根据质心决定当前元素属于哪个簇
    di=math.sqrt((data[i][0]-c[0][0])*(data[i][0]-c[0][0])+(data[i][1]-c[0][1])*(data[i][1]-c[0][1]))
    m=0
    for j in range(1,k):
        dc=math.sqrt((data[i][0]-c[j][0])*(data[i][0]-c[j][0])+(data[i][1]-c[j][1])*(data[i][1]-c[j][1]))
        if(dc<di):
            di=dc
            m=j
    mask1[m].append(data[i])        

oldVar=-1
newVar=getVar(mask1,c)
while(abs(newVar-oldVar)>=1):
    c=getMeans(mask1,c)
    oldVar=newVar
    newVar=getVar(mask1,c)
    mask1=[[] for i in range(k)]
    for i in range(0,k):
        mask1[i].append([0,0])
    for i in range(0,l):#根据质心决定当前元素属于哪个簇
        di=math.sqrt((data[i][0]-c[0][0])*(data[i][0]-c[0][0])+(data[i][1]-c[0][1])*(data[i][1]-c[0][1]))
        m=0
        for j in range(1,k):
            dc=math.sqrt((data[i][0]-c[j][0])*(data[i][0]-c[j][0])+(data[i][1]-c[j][1])*(data[i][1]-c[j][1]))
            if(dc<di):
               di=dc
               m=j
        mask1[m].append(data[i])         

for i in range(0,len(c)):
    colors=randRGB()
    if(len(mask1[i])>1):
        for j in range(1,len(mask1[i])):
            plt.plot(mask1[i][j][0],mask1[i][j][1],'o',color=colors)
            plt.plot(c[i][0],c[i][1],"rx") 

plt.show()   