#coding=utf-8
#!/usr/bin/env python

# import numpy as np
# import struct
# import math
# import random
# import tempfile
# import string
# import cv2
import attr
# import operator
from readMnist import *
# import matplotlib.pyplot as plt
# from sklearn import datasets, decomposition
# from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
# from sklearn import metrics
import time



def pcaAlgo(dataMat, testMat,K=attr.Attr["K"]):
    # 减去平均数
    # 计算协方差矩阵
    # 计算协方差矩阵的特征值和特征向量
    # 将特征值从大到小排序
    # 保留最大的K个特征向量
    # 将数据转换到上述K各特征向量构建的新空间中
    meanVal = np.mean(dataMat, axis=0)
    TmeanVal = np.mean(testMat, axis=0)
    DelAVG = dataMat - meanVal           #减去平均值
    TDelAVG = testMat - TmeanVal  # 减去平均值
    covMat = np.cov(DelAVG, rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat)) #计算特征值和特征向量
    eigInd = np.argsort(eigVals)
    eigInd = eigInd[:-(K+1):-1]   #保留最大前K个特征
    EigVec = eigVects[:,eigInd]        #对应的特征向量
    resImgs = DelAVG * EigVec     #将数据转换到低维新空间
    testImgs = TDelAVG * EigVec
    return resImgs, testImgs


def knn(resImgs,trainLabels,testImgs,k = attr.Attr["K"]):
    result = []
    # cnt = 0
    for img in testImgs:
        # print(cnt)
        # cnt += 1
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

    # PCA
    resImgs,testImgs = pcaAlgo(resImgs,testImgs)

    #取消注释以选择部分训练集
    select = 200
    resImgs = resImgs[0:select]
    trainLabels = trainLabels[0:select]

    T1 = time.time()
    testResult = knn(resImgs, trainLabels, testImgs)
    T2 = time.time()
    print('knn calc time :', T2 - T1, ' second', '\n')
    score = accuracy_score(testLabels, testResult)
    print("The accruacy is : ", score)
