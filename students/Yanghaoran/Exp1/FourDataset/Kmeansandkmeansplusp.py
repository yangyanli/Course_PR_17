'''
 It is implemented accoring to kmeans & kmeans++ algorithm
'''


import numpy as np
import matplotlib.pyplot as plt

def ReadFile(filename):
    fr=open(filename)
    data=[]
    for line in fr.readlines():
        modiline=line.strip().split(',')
        x=[float(modiline[0]),float(modiline[1])]
        data.append(x)

    return data

'''
随机选择初始点是kmeans所用的方法
'''
def Rand_Center(k,data):
    center=np.zeros((k,2))
    n=len(data)
    index=np.random.randint(n,size=k)
    for i in range(k):
        center[i]=data[index[i]]

    return center
'''
这个是根据kmeans++提供的算法来选择初始点
'''
def KpCenter(k,data):
    n=len(data)
    index=np.random.randint(n)
    center=np.zeros((k,2))
    center[0]=data[index]

    for i in range(1,k):
        D = np.zeros(n)
        for j in range(n):
            dis=np.inf
            for l in range(0,i):
                d=np.sum(np.power(np.array(data[j])-center[l],2))
                if d<dis:
                    dis=d
            D[j]=dis
        maxn=0
        for p in range(len(D)):
            if(D[p]>maxn):
                maxn=p
        center[i]=data[maxn]
    return center
def CalCluster(center,data):
    k=len(center)
    n=len(data)
    cluster=[[] for i in range(k)]
    for i in range(n):
        min=-1
        distance=np.inf
        for j in range(k):
            d=np.sum(np.power(center[j]-np.array(data[i]),2))
            if d<distance:
                min=j
                distance=d

        cluster[min].append(data[i])
    newcenter=np.zeros((k,2))
    #newcenter=list(np.mean(X,axis=0)) for X in cluster]
    for i in range(k):
        newcenter[i]=np.mean(cluster[i],axis=0)

    return cluster,newcenter




def Kmeans(k,data,alg="1"):
   # center=Rand_Center(k,data)
    if(int(alg)-2==0):
        center=KpCenter(k,data)
        print("kmeans+=")
    else:
        center=Rand_Center(k,data)

    cluster,newcenter=CalCluster(center,data)
    iter=1
    while not np.array_equal(center,newcenter):
        center=newcenter

        cluster,newcenter=CalCluster(center,data)
        print(iter,": ")
        iter+=1
    return cluster,newcenter

def Plot(cluster):
    cluster=np.array(cluster)
    color = ['black','gray','red','darkgreen','blue','m','crimson','lime','coral','snow','yellow','teal','lightpink','orange',
             'peru','blueviolet','skyblue']
    fig, ax = plt.subplots()
    k=len(cluster)
    for i in range(k):
        array = np.array(cluster[i])
        ax.scatter(array[:, 0], array[:, 1], c=color[i],
                   alpha=0.3)

    # ax.legend()
    ax.grid(True)
    plt.show()

'''
1 kmeans,2 kmeans++
'''
def runkmeans(alg="1"):
    data = ReadFile("DATA/mix.txt")

    cluster, newcenter = Kmeans(17, data,alg)

    Plot(cluster)



if __name__ == '__main__':
    runkmeans("2")

