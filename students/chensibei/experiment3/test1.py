#################################################
# SVM: support vector machine
# Author : zouxy
# Date   : 2013-12-12
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import numpy as np
import svm3
from sklearn.decomposition import PCA
import tensorflow as tf

################## test svm #####################
## step 1: load data
print "step 1: load data..."
#dataSet = []
#labels = []
#fileIn = open('E:/Python/Machine Learning in Action/testSet.txt')
#for line in fileIn.readlines():
#    lineArr = line.strip().split('\t')
#    dataSet.append([float(lineArr[0]), float(lineArr[1])])
#    labels.append(float(lineArr[2]))

#dataSet = mat(dataSet)
#labels = mat(labels).T
pca = PCA(n_components=5)


batch = mnist.train.next_batch(1000)
train_x = batch[0]
train_y = batch[1]

train_x = pca.fit_transform(train_x)

y = np.argmax(train_y,1)
result0 = []
result1 = []
result2 = []
result3 = []
result4 = []
result5 = []
result6 = []
result7 = []
result8 = []
result9 = []

temp1 = []
temp1.append(1)
temp_1=[]
temp_1.append(-1)
    #y = tf.argmax(train_y[0])
for j in  range(1000):
    if(y[j]==0):
        result0.append(temp1)
    else:
        result0.append(temp_1)

C = 11
toler = 0.01
maxIter = 10


    #print train_x.shape[0]
    #print train_x.shape[1]
    #print result0

svmClassifier0 = svm3.trainSVM(train_x, result0, C, toler, maxIter, kernelOption=('rbf', 1.0))
print "trained :0"
svmClassifier1 = svm3.trainSVM(train_x, result1, C, toler, maxIter, kernelOption=('rbf', 1.0))
print "trained :1"
svmClassifier2 = svm3.trainSVM(train_x, result2, C, toler, maxIter, kernelOption=('rbf', 1.0))
print "trained :2"
svmClassifier3 = svm3.trainSVM(train_x, result3, C, toler, maxIter, kernelOption=('rbf', 1.0))
print "trained :3"
svmClassifier4 = svm3.trainSVM(train_x, result4, C, toler, maxIter, kernelOption=('rbf', 1.0))
print "trained :4"
svmClassifier5 = svm3.trainSVM(train_x, result5, C, toler, maxIter, kernelOption=('rbf', 1.0))
print "trained :5"
svmClassifier6 = svm3.trainSVM(train_x, result6, C, toler, maxIter, kernelOption=('rbf', 1.0))
print "trained :6"
svmClassifier7 = svm3.trainSVM(train_x, result7, C, toler, maxIter, kernelOption=('rbf', 1.0))
print "trained :7"
svmClassifier8 = svm3.trainSVM(train_x, result8, C, toler, maxIter, kernelOption=('rbf', 1.0))
print "trained :8"
svmClassifier9 = svm3.trainSVM(train_x, result9, C, toler, maxIter, kernelOption=('rbf', 1.0))
print "trained :9"


accuracy0=0
for i in range(1000):
    batch = mnist.train.next_batch(1)
    test_x = batch[0]
    test_y = batch[1]
    test_x = pca.transform(test_x)
   # print test_x
    y = np.argmax(test_y, 1)
    result0 = []

    temp1 = []
    temp1.append(1)
    temp_1 = []
    temp_1.append(-1)
    # y = tf.argmax(train_y[0])
    for j in range(1):
        if (y[j] == 4):
            result0.append(temp1)
        else:
            result0.append(temp_1)

 #   print test_x
    kk=20;

    accuracy0, predict0 = svm3.testSVM( svmClassifier0 ,test_x, result0,kk+5)
    if(accuracy0==1 and not predict0 == 0):
        accuracy1, predict1 = svm3.testSVM(svmClassifier1, test_x, result0, kk + 5)
        if (accuracy1 == 1 and not predict1 == 0):
            accuracy2, predict2 = svm3.testSVM(svmClassifier2, test_x, result0, kk + 5)
            if (accuracy2 == 1 and not predict2 == 0):
                accuracy3, predict3 = svm3.testSVM(svmClassifier3, test_x, result0, kk + 5)
                if (accuracy3 == 1 and not predict3 == 0):
                    accuracy4, predict4 = svm3.testSVM(svmClassifier4, test_x, result0, kk + 5)
                    if (accuracy4 == 1 and not predict4 == 0):
                        accuracy5, predict5 = svm3.testSVM(svmClassifier5, test_x, result0, kk + 5)
                        if (accuracy5 == 1 and not predict5 == 0):
                            accuracy6, predict6 = svm3.testSVM(svmClassifier6, test_x, result0, kk + 5)
                            if (accuracy6 == 1 and not predict6 == 0):
                                accuracy7, predict7 = svm3.testSVM(svmClassifier7, test_x, result0, kk + 5)
                                if (accuracy7 == 1 and not predict7 == 0):
                                    accuracy8, predict8 = svm3.testSVM(svmClassifier8, test_x, result0, kk + 5)
                                    if (accuracy8 == 1 and not predict8 == 0):
                                        accuracy9, predict9 = svm3.testSVM(svmClassifier9, test_x, result0, kk + 5)
                                        ac += accuracy9
                                        continue
                                else:
                                    ac += accuracy7
                                    pre = bytes(predict7)
                                    print "predict:" + pre
                                    continue
                            else:
                                ac += accuracy6
                                pre = bytes(predict6)
                                print "predict:" + pre
                                continue
                        else:
                            ac += accuracy5
                            pre = bytes(predict5)
                            print "predict:" + pre
                            continue
                    else:
                        ac += accuracy4
                        pre = bytes(predict4)
                        print "predict:" + pre
                        continue
                else:
                    ac += accuracy3
                    pre = bytes(predict3)
                    print "predict:" + pre
                    continue
            else:
                ac += accuracy2
                pre = bytes(predict2)
                print "predict:" + pre
                continue
        else:
            ac += accuracy1
            pre = bytes(predict1)
            print "predict:" + pre
            continue
    else:
        ac+=accuracy0
        pre = bytes(predict0)
        print "predict:" + pre
        continue


    if(predict0==result0):
        accuracy0+=1;

ac=bytes(accuracy0)
print "result:"+ac

