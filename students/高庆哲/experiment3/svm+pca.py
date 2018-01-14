from tensorflow.examples.tutorials.mnist import input_data
import  sklearn.svm
import  numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import operator
from pca_knn import pca
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

print('00')
traindata = mnist.train.images
trainlabel = mnist.train.labels

testdata = mnist.test.images
testlabel = mnist.test.labels

del mnist


all = np.concatenate((traindata ,testdata))

all = pca(all ,40)[0].real
all = np.array(all)
pcadata = all[0:55000]
pcatest = all[55000:]


my = sklearn.svm.SVC()

my.fit(pcadata,trainlabel)

y_hat = my.predict(pcatest)

print(np.sum(y_hat == testlabel) / testlabel.shape[0])