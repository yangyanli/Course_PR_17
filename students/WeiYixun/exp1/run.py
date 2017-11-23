# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from kmeans import KMeansClassifier
from kmeans import biKMeansClassifier
import matplotlib.pyplot as plt

#加载数据集，DataFrame格式，最后将返回为一个matrix格式
def loadDataset(infile):
    df = pd.read_csv(infile, sep=',', header=0, dtype=str, na_filter=False)
    return np.array(df).astype(np.float)

if __name__=="__main__":
    data_X = loadDataset(r"data/Aggregation.txt")
    k = 7
    clf = biKMeansClassifier(k)
    clf.fit(data_X)
    cents = clf._centroids
    labels = clf._labels
    sse = clf._sse
    colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
    for i in range(k):
        index = np.nonzero(labels==i)[0]
        x0 = data_X[index, 0]
        x1 = data_X[index, 1]
        y_i = i
        for j in range(len(x0)):
            plt.text(x0[j], x1[j], str(y_i), color=colors[i], \
                        fontdict={'weight': 'bold', 'size': 6})
        plt.scatter(cents[i,0],cents[i,1],marker='x',color=colors[i],\
                    linewidths=7)
    
    plt.title("SSE={:.2f}".format(sse))
    plt.axis([0,40,0,30])
    #plt.axis([0,30,0,30])
    #outname = "./result/k_clusters" + str(k) + ".png"
    outname = "./result/bi_Aggregation.png"
    plt.savefig(outname)
    plt.show()
    
    
