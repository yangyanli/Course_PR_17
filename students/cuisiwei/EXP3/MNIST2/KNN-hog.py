#coding=utf-8
#!/usr/bin/env python

import numpy as np
import struct
import math
import random
import tempfile
import string
import cv2
import attr
from readMnist import *
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

#获取HOG特征
def hogFeature(data):
    hogF = []
    hog = cv2.HOGDescriptor('hog.xml')
    for img in data:
        img = np.reshape(img,(28,28))#重新恢复图像
        cv_img = img.astype(np.uint8)
        hog_feature = hog.compute(cv_img)
        hogF.append(hog_feature)
    # print("hogF:",np.shape(hogF))
    hogF = np.array(hogF)
    hogF = np.reshape(hogF,(-1,attr.Attr["hogDim"]))
    return hogF

def knn(resImgs,trainLabels,testImgs,k = attr.Attr["K"]):
    result = []
    cnt = 0
    for img in testImgs:
        print(cnt)
        cnt += 1
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

    #HOG

    resImgs = hogFeature(trainImgs)
    testImgs = hogFeature(testImgs)
    # print(np.shape(testImgs))

    # #选择部分训练集
    # select = 2000
    # resImgs = resImgs[0:select]
    # trainLabels = trainLabels[0:select]

    testResult = knn(resImgs,trainLabels,testImgs)

    # print(np.shape(testResult))
    # print(type(testResult))
    # print(testResult)
    score = accuracy_score(testLabels,testResult)
    print("The accruacy is : ", score)

