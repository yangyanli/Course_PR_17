#coding=utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

#下面实现k-means对数据进行聚类分析

#这个函数用于计算两点之间的欧式距离
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# 随机生成k个质心
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape # 得到dataSet的大小
    centroids = zeros((k, dim)) # 建立k行，dim列的零矩阵 存储k个质心的坐标
    for i in range(k):
        index = int(random.uniform(0, numSamples)) # 生成0-numSamples中的一个随机行数
        centroids[i, :] = dataSet[index, :] # 从dataset中找到随机数对应的位置坐标
    return centroids


# 具体实现k-means方法
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = mat(zeros((numSamples, 2))) # 第一列用于存储这个坐标点属于哪个cluster，第二列用于存储这个点到它所在cluster质心的距离
    clusterChanged = True # 初始化clusterChanged为true，用于是否更改了质心

    # 初始化质心，调用上面的函数
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        # 遍历每一个点
        for i in xrange(numSamples):
            minDist = 100000.0
            minIndex = 0
            # 对于每一个质心来说
            # 找到每个点距离最近的质心，并用minIndex记录其归属的质心是第几个
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
                    # 更新点所在的cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2 #距离别忘记平方

        #更新质心
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            #pointsIncluster是一个矩阵，内部存储的是聚类到这个cluster的所有点
            # ==j判断是否属于这个cluster，如果属于则为1，取出它的下标，即行号nonzero(clusterAssment[:, 0].A == j)，后面的[0]是取矩阵的第0列
            centroids[j, :] = mean(pointsInCluster, axis=0) # 压缩行，对各列求均值，用均值的位置来作为新的质心位置
            # print centroids

    print 'K-means聚类完成'
    return centroids, clusterAssment


# 将结果用散点图表示出来
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print "请确保数据是二维数据"
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print "K值过大，请重新选取"
        return 1

        # 画出所有点
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画出k个质心
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


if __name__=="__main__":
    file = open('D://R15.txt')#可以换成其他的三个图进行测试
    lines = file.readlines()

    x = []
    y = []
    for line in lines:
        temp = line.replace('\n', '').split(',')
        x.append(float(temp[0]))  # 数据一定要类型转化，否则默认是string类型读入，画图会出错
        y.append(float(temp[1]))

    # 将数据可视化一下，看看总体情况
    f1 = plt.figure(1)
    # plt.subplot(211) subplot函数用于显示多个子图，2表示总共两行，1表示总共一列，1表示显示在第一个位置
    a = plt.scatter(x, y)

    plt.show(a)

    ## 第一步：读入txt的数据
    print "1.从txt中读取数据"
    dataSet = []
    for line in lines:
        datas = line.replace('\n', '').split(',')
        dataSet.append([float(datas[0]), float(datas[1])])#一维数组，每一维都存了两个数据
    print dataSet
    ## step 2: 聚类
    print "2.聚类"
    dataSet = mat(dataSet) # 将数组变成一个矩阵，每一行存一个点
    print dataSet
    k = 8
    centroids, clusterAssment = kmeans(dataSet, k)

    ## step 3: 结果
    print "3.结果"
    showCluster(dataSet, k, centroids, clusterAssment)









