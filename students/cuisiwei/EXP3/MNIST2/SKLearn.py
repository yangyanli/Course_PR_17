#coding=utf-8
#!/usr/bin/env python

import numpy as np
import struct
import math
import random
import tempfile
import string
import cv2
import time

import attr
import operator
from readMnist import *
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import metrics


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


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model

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

    print("without HOG:")

    model = naive_bayes_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of naive_bayes_classifier is : ", score)

    model = knn_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of knn_classifier is : ", score)


    model = logistic_regression_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of logistic_regression_classifier is : ", score)


    model = random_forest_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of random_forest_classifier is : ", score)



    model = decision_tree_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of decision_tree_classifier is : ", score)


    model = gradient_boosting_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of gradient_boosting_classifier is : ", score)



    #HOG

    resImgs = hogFeature(trainImgs)
    testImgs = hogFeature(testImgs)
    # print(np.shape(feature))

    #选取部分训练数据
    # select = 500
    # resImgs = resImgs[0:select]
    # trainLabels = trainLabels[0:select]

    print("adding HOG feature:")

    model = naive_bayes_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of naive_bayes_classifier is : ", score)


    model = knn_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of knn_classifier is : ", score)


    model = logistic_regression_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of logistic_regression_classifier is : ", score)


    model = random_forest_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of random_forest_classifier is : ", score)



    model = decision_tree_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of decision_tree_classifier is : ", score)


    model = gradient_boosting_classifier(resImgs,trainLabels)
    testResult = model.predict(testImgs)
    score = accuracy_score(testLabels, testResult)
    print("The accruacy of gradient_boosting_classifier is : ", score)

