#coding=utf-8
import random
import math
import struct
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt

  
def loadImage(filename):  
  
    binfile = open(filename, 'rb') # 读取二进制文件  
    buffers = binfile.read()  
  
    head = struct.unpack_from('>IIII', buffers, 0) # 取前4个整数，返回一个元组  
  
    offset = struct.calcsize('>IIII')  # 定位到data开始的位置  
    imgNum = head[1]  
    width = head[2]  
    height = head[3]  
  
    bits = imgNum * width * height  # 一共有 60000*28*28个像素值  
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'  
  
    imgs = struct.unpack_from(bitsString, buffers, offset) # 取data数据，返回一个元组  
  
    binfile.close()  
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshape为[60000,784]型数组  
  
    return imgs  
  
  
def loadLabel(filename):  
  
    binfile = open(filename, 'rb') # 读二进制文件  
    buffers = binfile.read()  
  
    head = struct.unpack_from('>II', buffers, 0) # 取label文件前2个整形数  
  
    labelNum = head[1]  
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置  
  
    numString = '>' + str(labelNum) + "B" # fmt格式：'>60000B'  
    labels = struct.unpack_from(numString, buffers, offset) # 取label数据  
  
    binfile.close()  
    labels = np.reshape(labels, [labelNum]) # 转型为列表(一维数组)  
  
    return labels 

class kmeans():
    label_num = 10
    data_center = [] 
    data_loc = []
    data_ground_truth = []
    data_label = []
    data_num = 0
    error_ = 0.05
    final_accuary = 0
    length = 0
    def __init__(self,data,label):
        self.data_loc = data
        self.data_ground_truth = label
        self.data_num = int(len(data))
        self.length = len(self.data_loc[0])
        self.data_center = np.zeros([self.label_num,self.length], dtype=np.float32)
        self.data_label = np.zeros([len(data)])

    def euclid_distance(self,x1,x2):
        return np.sqrt(np.sum(x1-x2)**2)


    def center_init(self):#k-means++
        first = random.randint(0,self.data_num-1)
        self.data_center[0,:] = self.data_loc[first,:]
        data_distance = [] 
        sum = 0
        for i in range(self.data_num):
            data_distance.append(self.euclid_distance(self.data_loc[i,:],self.data_center[0,:]))
            sum += data_distance[i]

        for i in range(1,self.label_num,1):
            random_sum = random.uniform(0,sum)
            for j in range(self.data_num):
                random_sum -= data_distance[j]
                if random_sum <= 0:
                    self.data_center[i,:] = self.data_loc[j,:]
                    break

    def find_nearest_center(self,x):
        min_distance = float("inf")
        index = 0
        for i in range(self.label_num):
            distance = self.euclid_distance(x,self.data_center[i,:])
            if distance < min_distance:
                min_distance = distance
                index = i
        return index,min_distance



    def center_update(self):
        while True:
            center_sum_loc = np.zeros([self.label_num,self.length],dtype=np.float32)
            center_num = np.zeros([self.label_num],dtype=np.int32)
            change_num = 0
            for i in range(self.data_num):
                center,distance = self.find_nearest_center(self.data_loc[i,:])
                center_sum_loc[center,:] += self.data_loc[i,:]
                center_num[center] += 1
                if(self.data_label[i]!=center):
                    change_num += 1
                    self.data_label[i]=center

            if change_num *1.0 / self.data_num < self.error_:
                break

            for i in range(self.label_num):
                for j in range(self.length):
                    self.data_center[i,j] = center_sum_loc[i,j]/center_num[i]

    def calc_accuracy(self):
        
        self.fianl_accuary = adjusted_mutual_info_score(self.data_label, self.data_ground_truth)
        print "all : {} , accuary : {} \n".format(self.data_num, self.fianl_accuary)

    def data_output(self):
        
        file = open(self.file_out,"w")
        for i in range(self.data_num):
            file.write("{},{},{}\n".format(self.data_loc[i,0],self.data_loc[i,1],self.data_label[i]))  
  
if __name__ == "__main__":  
    file1= 'data/MNIST/train-images-idx3-ubyte'  
    file2= 'data/MNIST/train-labels-idx1-ubyte'  
    print "loading data..."
    data = loadImage(file1)
    #print (len(data))
    #data = data[0:10000,:]   
    #print(len(data))  
    # plt.imshow(data[7,:].reshape(28,28),cmap = 'binary')   
    # plt.show()
    label = loadLabel(file2)    
    #label = label[0:10000]
    
    k = kmeans(data,label)
    k.center_init()
    k.center_update()
    k.calc_accuracy()
    
