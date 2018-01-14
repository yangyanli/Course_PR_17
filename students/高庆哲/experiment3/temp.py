
from tensorflow.examples.tutorials.mnist import input_data
import  numpy as np

import operator

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)


print('00')
traindata = mnist.train.images
trainlabel = mnist.train.labels

testdata = mnist.test.images
testlabel = mnist.test.labels

del mnist


# 零均值化
def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal


def pca(dataMat, n):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
    return lowDDataMat, reconMat

def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

def pca_per(dataMat,percentage=0.99):
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    n=percentage2n(eigVals,percentage)                 #要达到percent的方差百分比，需要前n个特征向量
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据
    return lowDDataMat,reconMat

def classify(Inx, Dataset, labels, k):
    DataSetSize = Dataset.shape[0]  # 获取数据的行数，shape[1]位列数
    diffmat = np.tile(Inx, (DataSetSize, 1)) - Dataset
    SqDiffMat = diffmat**2
    SqDistances = SqDiffMat.sum(axis=1)
    Distance = SqDistances**0.5
    SortedDistanceIndicies = Distance.argsort()
    ClassCount = {}
    for i in range(k):
        VoteiLabel = labels[SortedDistanceIndicies[i]]
        ClassCount[VoteiLabel] = ClassCount.get(VoteiLabel, 0) + 1
    SortedClassCount = sorted(ClassCount.items(), key = operator.itemgetter(1), reverse = True)
    return SortedClassCount[0][0]

def knn(test, data, label,k):

    pre_label = []
    count = 0
    for pic in test:
        count+=1
        if count%50==0:
            print(count)
        detamatrix = data-pic
        dsquare = np.square(detamatrix)

        sum_ds = np.sum(dsquare,axis=1)

        distance = sum_ds**0.5

        sorteddis = distance.argsort()

        classCount = {}
        for i in range(k):
            voteLabel = label[sorteddis[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

        pre_label.append(sortedClassCount[0][0])





    return pre_label





def testa():
    a = np.array([[ 1.  ,  1.  ,7],
       [ 0.9 ,  0.95,5],
       [ 1.01,  1.03,6],
       [ 2.  ,  2.  ,6],
       [ 2.03,  2.06,8],
       [ 1.98,  1.89,33],
       [ 3.  ,  3.  ,36],
       [ 3.03,  3.05,5],
       [ 2.89,  3.1 ,78],
       [ 4.  ,  4.  ,36],
       [ 4.06,  4.02,22],
       [ 3.97,  4.01,36]])

    print(a)
    print(pca_per(a))



if __name__ == '__main__':
    # result = knn(testdata[0:1000],traindata,trainlabel,10)
    # np.save('result/pureknn',result)
    result = np.load('result/pureknn.npy')
    all_count = 0
    rightcount = 0
    print(result)

    for i in range(len(result)):

        if result[i] == testlabel[i]:
            rightcount +=1
        all_count+=1

    print(rightcount/all_count)