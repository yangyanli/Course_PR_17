import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import  sklearn.svm
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
import numpy as np

print('00')
traindata = mnist.train.images
trainlabel = mnist.train.labels

testdata = mnist.test.images
testlabel = mnist.test.labels

del mnist

print('0')

my = sklearn.svm.SVC()

my.fit(traindata,trainlabel)

y_hat = my.predict(testdata)

print(np.sum(y_hat == testlabel) / testlabel.shape[0])

