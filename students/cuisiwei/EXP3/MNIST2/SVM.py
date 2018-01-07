#coding=utf-8
#!/usr/bin/env python
#
# import numpy as np
# import struct
# import math
# import random
# import tempfile
# import string
import cv2
import attr
# import operator
from readMnist import *
# import matplotlib.pyplot as plt
# from sklearn import datasets, decomposition
# from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA

def svm_baseline():
    # 加载数据
    training_data = resImgs
    test_data = testImgs
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    # model = OneVsRestClassifier(svm.SVC(kernel='poly'))

    clf = model.fit(training_data, trainLabels)
    # print(clf.score(training_data, trainLabels))
    print(clf.score(test_data, testLabels))


#获取HOG特征
def hogFeature(data):
    features = []
    hog = cv2.HOGDescriptor('hog.xml')
    for img in data:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)
        hog_feature = hog.compute(cv_img)
        features.append(hog_feature)
    features = np.array(features)
    features = np.reshape(features,(-1,attr.Attr["hogDim"]))
    return features


def PCACALC(data,test,K = attr.Attr["K"]):
    # print(data)
    pca = PCA(n_components=K,copy=False)
    # pca = decomposition.PCA()
    pca.fit(data)
    # print(np.shape(data))
    resImgs = pca.fit_transform(data)
    # print(np.shape(data))
    # print(data[0])
    # print(np.shape(newData))
    # print(newData[0])
    testImgs = pca.transform(test)
    # plt.figure()
    # plt.plot(pca.explained_variance_, 'k', linewidth=2)
    # plt.xlabel('n_components', fontsize=16)
    # plt.ylabel('explained_variance_', fontsize=16)
    # plt.show()
    return  resImgs,testImgs


if __name__ == "__main__":

    #加载图像和标签
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
    # resImgs, testImgs = PCACALC(resImgs,testImgs)

    #HOG
    #取消注释选择提取HOG特征
    resImgs = hogFeature(trainImgs)
    testImgs = hogFeature(testImgs)

    #取消注释选择一定的训练集
    select = 2000
    resImgs = resImgs[0:select]
    trainLabels = trainLabels[0:select]

    svm_baseline()




