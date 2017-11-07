#coding:utf-8
import random
import math
import copy
import matplotlib.pyplot as plt
class Node:
    def __init__(self,x,y,lable):
        self.x=x
        self.y=y
        self.lable=lable
        self.neighbor=[]
#点的集合
nodes=[]
#中心点的集合
m_nodes=[]
def initNode(filepath):
    global nodes
    file=open(filepath,'r')
    line=file.readline()
    row=line.split(",")
    while len(line)!=0:
        print row[0],row[1]
        node=Node(float(row[0]),float(row[1]),0)
        nodes.append(node)
        line=file.readline()
        row=line.split(",")
        
def initM_Node(k):
    #k-mean++的方法初始化中心点
    global nodes
    length=len(nodes)
    #初始点随机选取
 #   m_node=Node(random.randint(1,20),random.randint(1,20),0)
    m_node=Node(10,24,0)
    m_nodes.append(m_node)
    for i  in range(k-1):
        temp=copy.deepcopy(nodes)
        D=[]
        for j in range(length):   
            m_temp=copy.deepcopy(m_nodes)
            node=temp.pop()           
            #计算和最近一个种子点的距离
            t=0
            while len(m_temp)!=0:
                m_node=m_temp.pop()
                t1=math.sqrt((node.x-m_node.x)*(node.x-m_node.x)+(node.y-m_node.y)*(node.y-m_node.y))
                if t==0:
                    t=t1
                else:
                    if t1<t:
                        t=t1
            D.append(t)
        sum_D=sum(D)
        rand=random.random()*sum_D
        for a in range(length):
            rand-=D[a]
            if rand<=0:
                m_node1=Node(nodes[a].x,nodes[a].y,0)
                m_nodes.append(m_node1)
                break        
        
            
def cluster(k):
    global nodes,m_nodes
    length_node=len(nodes)
    
    temp_nodes=copy.deepcopy(nodes)
    for q in range(30):
        temp_m_nodes=copy.deepcopy(m_nodes)
        #第一次循环找每个点的标签（所属的中心点）
        for i in range(length_node):
            dist=0 
            for j in range(k):
                new_dist=math.sqrt((temp_m_nodes[j].x-temp_nodes[i].x)*(temp_m_nodes[j].x-temp_nodes[i].x)+(temp_m_nodes[j].y-temp_nodes[i].y)*(temp_m_nodes[j].y-temp_nodes[i].y))
                if dist==0:
                    dist=new_dist
                    nodes[i].lable=j
                else:
                    if new_dist<dist:
                        dist=new_dist
                        nodes[i].lable=j
                        
         #               print "lable:"+str(nodes[i].lable)
            m_nodes[nodes[i].lable].neighbor.append(nodes[i])
        #更新中心点到他们的中心
        dist=0
        for p in range(k):
            dx=0
            dy=0
            num=len(m_nodes[p].neighbor)
            t_m_nodes=m_nodes[p].neighbor
            
            while len(t_m_nodes)!=0:
                node=t_m_nodes.pop()
                dx+=node.x
                dy+=node.y
            if num!=0:
                dist+=(m_nodes[p].x-dx/num)*(m_nodes[p].x-dx/num)+(m_nodes[p].y-dy/num)*(m_nodes[p].y-dy/num)
                m_nodes[p].x=dx/num
                m_nodes[p].y=dy/num
               
            else:
                dist+=(m_nodes[p].x)*(m_nodes[p].x)+(m_nodes[p].y)*(m_nodes[p].y)
                m_nodes[p].x=0
                m_nodes[p].y=0
        
        print "dist: "+str(dist)
        #show(k)      
        
def show(k):
    global nodes
    temp=copy.deepcopy(nodes)
    length=len(temp)
    m_nodes1=copy.deepcopy(m_nodes)
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
    color=["g.","b.","y.","y.","m+","c+","k+","g+","c.","k.",".y","m.","g.","b."]
    while len(cluster_x)!=0:
        x=cluster_x.pop()
        y=cluster_y.pop()
        plt.plot(x,y,color.pop())
    while len(m_nodes1)!=0:
        p1=m_nodes1.pop()
        plt.plot(p1.x,p1.y,'r.')
   
    plt.show()
            
if __name__=="__main__":
    k=7
    initNode("c:/Users/Bluepanda/Desktop/Aggregation.txt")
    initM_Node(k-1)
    cluster(k-1)
    show(k-1)
    
