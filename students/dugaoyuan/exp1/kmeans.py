from numpy import *  
import time  
import matplotlib.pyplot as plt  
    
def kmeans(dataSet, k):  
    numSamples = dataSet.shape[0]  
    numSamples, dim = dataSet.shape  
    centroids = zeros((k, dim))  
    clusterset = mat(zeros((numSamples, 2)))  
    
    for i in range(k):  
        index = int(random.uniform(0, numSamples))  
        centroids[i, :] = dataSet[index, :] 

    cluster = True  
    while cluster:  
        cluster = False  //标记
        for i in xrange(numSamples):  
            minDist  = 50000.0  
            minIndex = 0  
             
            for j in range(k):  
                distance = sqrt(sum(power(centroids[j, :], dataSet[i, :], 2))) 
                if distance < minDist:  
                    minDist  = distance  
                    minIndex = j  
             
            if clusterset[i, 0] != minIndex:  
                cluster = True  
                clusterset[i, :] = minIndex, minDist**2  
  
        for j in range(k):  
            pointsInCluster = dataSet[nonzero(clusterset[:, 0].A == j)[0]]  
            centroids[j, :] = mean(pointsInCluster, axis = 0)  
        print '+one time' 
    print 'Cluster complete!'  
    return centroids, clusterset  

def visualization(dataSet, k, centroids, clusterset):  
    numSamples, dim = dataSet.shape 
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr','or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr','or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):  
        print "Sorry! Your k is too large!"  
        return 1  
  
    for i in xrange(numSamples):  
        markIndex = int(clusterset[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12) 
    plt.show()  
