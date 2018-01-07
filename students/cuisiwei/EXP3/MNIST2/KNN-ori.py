#coding=utf-8
#!/usr/bin/env python

# import numpy as np
# import struct
# import math
# import random
# import tempfile
# import string
import attr
from readMnist import *
from sklearn.metrics import accuracy_score

#计算KNN，从Attr中取得参数K
def knn(resImgs,trainLabels,testImgs,k = attr.Attr["K"]):
    result = []
    #对每个测试数据
    for img in testImgs:
        knnList = []
        maxId = -1
        maxDist = 0
        #初始化，放K个
        for i in range(k):
            label = trainLabels[i]
            timg = resImgs[i]
            #计算两个点的欧氏距离
            dist = np.linalg.norm(timg - img)
            knnList.append((dist,label))
        #剩余训练集进行更新
        for i in range(k,len(trainLabels)):
            label = trainLabels[i]
            timg = resImgs[i]
            # 计算两个点的欧氏距离
            dist = np.linalg.norm(timg - img)
            if maxId<0:
                for j in range(k):
                    if maxDist<knnList[j][0]:
                        maxId = j;
                        maxDist = knnList[maxId][0]
            if dist<maxDist:
                knnList[maxId] = (dist,label)
                maxId = -1;
                maxDist = 0
        #保存分多少类
        labelType = attr.Attr["Type"]
        TypeCnt = [0 for i in range(labelType)]
        for dist,label in knnList:
            TypeCnt[label] += 1
        # 矩阵中最大元素的位置
        result.append(np.argmax(TypeCnt))
    return np.array(result)


if __name__ == "__main__":
    #读取数据
    trainImgs = loadImage()
    # print(imgs[0])
    trainLabels = loadLabel()
    # print(np.shape(labels))
    # print(int(labels[0]))

    trainLabels = np.asarray(trainLabels.flatten())

    testImgs = loadImage('mnist/t10k-images-idx3-ubyte')
    testLabels = loadLabel('mnist/t10k-labels-idx1-ubyte')
    testLabels = np.asarray(testLabels.flatten())

    # print(trainLabels[0])
    resImgs = trainImgs

    # 取消注释以选择部分训练集
    # select = 2000
    # resImgs = resImgs[0:select]
    # trainLabels = trainLabels[0:select]


    testResult = knn(resImgs,trainLabels,testImgs)
    score = accuracy_score(testLabels,testResult)
    print("The accruacy is : ", score)

