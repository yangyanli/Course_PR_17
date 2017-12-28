import numpy as np
import operator
import sklearn.datasets
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

def line2Mat(line):
    line = line.strip().split(' ')
    label = line[0]
    mat = []
    for pixel in line[1:]:
        pixel = pixel.split(':')[1]
        mat.append(float(pixel))
    return mat, label

def file2Mat(fileName):
    f = open(fileName)
    lines = f.readlines()
    matrix = []
    labels = []
    for line in lines:
        mat, label = line2Mat(line)
        matrix.append(mat)
        labels.append(label)
    print('Read file ' + str(fileName) + ' to matrix done!')
    return np.array(matrix), labels

def dict2list(dic:dict):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

def classify(mat, matrix, labels, k):
    diffMat = np.tile(mat, (np.shape(matrix)[0], 1)) - matrix
    diffMat = np.array(diffMat)
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistanceIndex = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistanceIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(dict2list(classCount), key=lambda x:x[0], reverse=True)
    return sortedClassCount[0][0]

def classifyFiles(trainMatrix, trainLabels, testMatrix, testLabels, K):
    rightCnt = 0
    for i in range(len(testMatrix)):
        if i % 100 == 0:
            print('num ' + str(i) + '. ratio: ' + str(float(rightCnt) / (i + 1)))
        label = testLabels[i]
        predictLabel = classify(testMatrix[i], trainMatrix, trainLabels, K)
        if label == predictLabel:
            rightCnt += 1
    return float(rightCnt) / len(testMatrix)

mnist = fetch_mldata('MNIST original', data_home='./')
mnist.keys()
X_train, X_test, y_train, y_test = train_test_split(mnist.data / 255.0, mnist.target, test_size=0.15, random_state=42)
trainMatrix, testMatrix, trainLabels, testLabels = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
K = 10
rightRatio = classifyFiles(trainMatrix, trainLabels, testMatrix, testLabels, 1)
print('classify right ratio:'+str(rightRatio))