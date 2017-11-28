import numpy as np
import matplotlib.pyplot as plt
import math
import time

UNCLASSIFIED = False
NOISE = 0

def loadDataSet(fileName, splitChar='\t'):
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            curline.pop()
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet

def eps_neighbor(a, b, eps):
    return (math.sqrt(np.power(a - b, 2).sum())) < eps

def region_query(data, pointId, eps):
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i)
    return seeds

def expand_cluster(data, Result, point, ID, eps, minPts):
    seeds = region_query(data, point, eps)
    if len(seeds) < minPts:
        Result[point] = NOISE
        return False
    else:
        Result[point] = ID
        for i in seeds:
            Result[i] = ID
        while len(seeds) > 0:
            currentpoint = seeds[0]
            currentseeds = region_query(data,currentpoint,eps)
            if len(currentseeds) >= minPts:
                for i in currentseeds :
                    if Result[i]==UNCLASSIFIED:
                        Result[i] = ID
                        seeds.append(i)
                    elif Result[i]==NOISE:
                        Result[i] = ID
                    else:
                        Result[i] = ID
            seeds = seeds[1:]
        return True

def dbscan(data, eps, minPts):

    ID = 1

    nPoints = data.shape[1]

    Result = [UNCLASSIFIED]*nPoints

    for point in range(nPoints):

        if Result[point]==UNCLASSIFIED:

            if expand_cluster(data,Result,point,ID,eps,minPts):

                ID=ID+1

    return Result,ID-1  

def plotFeature(data, clusters, clusterNum,datalist):
    nPoints = data.shape[1]
    matClusters = np.mat(clusters).transpose()
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'brown', 'purple', 'orange','red']
    ax = fig.add_subplot(111)
    for co in range(0,8):
        x=[]
        y=[]
        for i in range(nPoints):
            colorSytle = scatterColors[(clusters[i]) % len(scatterColors)]
            if ((clusters[i]) % len(scatterColors)) == co:
                x.append(datalist[0][i])
                y.append(datalist[1][i])
        ax.scatter(x,y,c=scatterColors[co],marker = 'o',s=50)


def main():
    dataSet = loadDataSet('R15.txt', splitChar=',')
    dataSet = np.mat(dataSet).transpose()
    clusters, clusterNum = dbscan(dataSet, 2, 2)
    print("cluster Numbers = ", clusterNum)
    nPoints = dataSet.shape[1]
    data = dataSet.tolist()
    
    plotFeature(dataSet, clusters, clusterNum,data)


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))
    plt.show()
