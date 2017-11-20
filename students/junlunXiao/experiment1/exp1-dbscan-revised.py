#coding:utf-8
import copy
import math
import matplotlib.pyplot as plt
import time
class Node:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.lable=0

#节点
nodes=[]
#邻接矩阵
dist=[]
#类别数量
k=1 
def initNode(filepath):
    #从文件读入点
    file=open(filepath,'r')
    line=file.readline()
    row=line.split(",")
    while len(line)!=0:
        node=Node(float(row[0]),float(row[1]))
        nodes.append(node)
        line=file.readline()
        row=line.split(",")
        
def initN(Eps):
    #初始化邻域
    length=len(nodes)
    
    for i in range(length):
        for j in range(length):
            if i==j:
                dist.append(0)
                continue
            distance=float(math.sqrt((nodes[i].x-nodes[j].x)*(nodes[i].x-nodes[j].x)+(nodes[i].y-nodes[j].y)*(nodes[i].y-nodes[j].y)))
            dist.append(distance)
                    
def cluster(Eps,MinPts):
    global k
    length=len(nodes)
    for i in range(length):
        node=nodes[i]
        if node.lable==0:
            #若该点还没被处理过
            T=getN(i,Eps)
            if len(T)<MinPts:
                nodes[i].lable=-1
            else:
                #该点为核心点
                nodes[i].lable=k
                visited=[]
                start=time.clock()
                while len(T)!=0:
                    node1=T.pop()
                    if node1 in visited :
                        continue
                    visited.append(node1)
                    index=nodes.index(node1)
                    if nodes[index].lable==-1 or nodes[index].lable==0:
                        nodes[index].lable=k
                    T1=getN(index,Eps)
                    if len(T1)>=MinPts:
                        #list和set的速度差距超级大,修改的地方
                        T.extend(T1)
                        T=set(T)
                        T=list(T)
                k=k+1
                end=time.clock()
                print "循环时间："+str(end-start)
def show():
    global k
    temp=copy.deepcopy(nodes)
    length=len(temp)
    cluster_x=[]
    cluster_y=[]
    for j in range(k):
        cluster_x.append(list())
        cluster_y.append(list())
    for i in range(k):
        while len(temp)!=0:
           node=temp.pop()
      #     print "lable1:"+str(node.lable)
           cluster_x[node.lable].append(node.x)
           cluster_y[node.lable].append(node.y)
    color=["g*","b*","y*","k*","r*","c*","m*","g+","b+","y+","k+","r+","c+","m+","g.","b.","y.","k.","r.","c.","m."]

    while len(cluster_x)!=0:
        x=cluster_x.pop()
        y=cluster_y.pop()
        plt.plot(x,y,color.pop(),)
    plt.show()

def getN(i,EPs):
    global nodes,dist
    length = len(nodes)
    N=[]
    for j in range(length):
        if i==j:
            continue
        if dist[i*length+j]<=EPs and (nodes[j].lable==0 or nodes[j]==-1):
            N.append(nodes[j])
    return N

if __name__=="__main__":
    #R15 Eps=0.5 14
    #Aggregation Eps=2 4
    #mix Eps=2 4
    #flame Eps=1.0 3
    Eps=2
    MinPts=1
    start=time.clock()
    initNode("c:/Users/Bluepanda/Desktop/mix.txt")
    initN(Eps)
    end=time.clock()
    print "初始化时间："+str(end-start)
    start=time.clock()
    cluster(Eps,MinPts)
    end=time.clock()
    print "聚类时间："+str(end-start)
    show()
    print "共有点： "+str(len(nodes)) 
    print "类别 ："+str(k)