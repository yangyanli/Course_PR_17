'''
A simple implemetation of DBSCAN according to the process provided
in http://www.cnblogs.com/pinard/p/6208966.html
'''
import numpy as np
import matplotlib.pyplot as plt

def ReadData(fileName, seperator='\t'):

   f=open(fileName)
   data=[]
   for line in f.readlines():
       ModiLine=line.strip().split(seperator)
       x=[float(ModiLine[0]),float(ModiLine[1])]
       data.append(x)
   data=np.array(data)
   return data

def distance(point1,point2):
    return np.sqrt(np.sum(np.power(point1-point2,2)))

def dbscan(data,eps,Minpts):
    n=data.shape[0]
    noise=[]
    k=1
    label=[0 for i in range(n)]
    for i in range(n):
        point=data[i]
        if(label[i]==0):
            neighbour=Getneighbour(data,point,eps)
            if len(neighbour)>=Minpts:
                label[i]=k
                for j in neighbour:
                    label[j]=k
                while(len(neighbour)>0):
                    p=neighbour[0]
                    neighbour2=Getneighbour(data,data[p],eps)
                    if(len(neighbour2)>=Minpts):
                        for l in range(len(neighbour2)):
                            w=neighbour2[l]
                            if(label[w]==0):
                                neighbour.append(w)
                                label[w]=k
                    neighbour=neighbour[1:]
                k=k+1
    return label


def Getneighbour(data,point,eps):
    neighbour=[]
    for i in range(len(data)):
        if(distance(data[i],point)<=eps):
            neighbour.append(i)
    return neighbour


def plot(label,data):
    n=len(data)
    fig, ax = plt.subplots()
    color = ['red', 'green', 'blue', 'black', 'yellow', 'orange', 'purple', 'whitesmoke', 'c', 'magenta'
        , 'darkviolet', 'greenyellow', 'lime', 'brown', 'lightpink', 'saddlebrown', 'olive']
    for i in range(n):
        ax.scatter(data[i][0], data[i][1], c=color[label[i]],
                   alpha=0.3)

    ax.grid(True)
    plt.show()

if __name__ == '__main__':
    data=ReadData("DATA/mix.txt",',')  #Aggregation 2,15 //1.2 ,8//0.5,15 // 2,15//2.5,25
    label=dbscan(data,2.5,25)
    plot(label,data)
