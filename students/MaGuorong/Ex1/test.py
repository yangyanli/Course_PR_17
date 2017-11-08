# -*- coding: utf-8-*-
import matplotlib.pyplot as plt
import random
from K_meansplus import K_means_plus_plus
from kmeans import k_means


def main():
    points = []
    xs = []
    ys = []
    filein = open("Aggregation1.txt")
    for line in filein.readlines():
        temp = []
        lineArr = line.strip().split('\t')
        temp.append(float(lineArr[0]))
        temp.append(float(lineArr[1]))

        xs.append(float(lineArr[0]))
        ys.append(float(lineArr[0]))
        points.append(temp)
    filein.close()
    test = K_means_plus_plus(points, 7)
    centroids = test.final_centroids()

    nexttest = k_means(points, centroids, 7)
    clusters = nexttest.final_clusters()
    centroids = nexttest.final_centroids()
    group0x = []
    group0y = []
    group1x = []
    group1y = []
    group2x = []
    group2y = []
    group3x = []
    group3y = []
    group4x = []
    group4y = []
    group5x = []
    group5y = []
    group6x = []
    group6y = []
    centroidsx = []
    centroidsy = []

    for points in clusters[0]:
        group0x.append(points[0])
        group0y.append(points[1])
    for points in clusters[1]:
        group1x.append(points[0])
        group1y.append(points[1])
    for points in clusters[2]:
        group2x.append(points[0])
        group2y.append(points[1])
    for points in clusters[3]:
        group3x.append(points[0])
        group3y.append(points[1])
    for points in clusters[4]:
        group4x.append(points[0])
        group4y.append(points[1])
    for points in clusters[5]:
        group5x.append(points[0])
        group5y.append(points[1])
    for points in clusters[6]:
        group6x.append(points[0])
        group6y.append(points[1])
    for points in centroids:
        centroidsx.append(points[0])
        centroidsy.append(points[1])

    plt.scatter(group0x, group0y, color='red')
    plt.scatter(group1x, group1y, color='yellow')
    plt.scatter(group2x, group2y, color='blue')
    plt.scatter(group3x, group3y, color='orange')
    plt.scatter(group4x, group4y, color='pink')
    plt.scatter(group5x, group5y, color='green')
    plt.scatter(group6x, group6y, color='gray')

    plt.scatter(centroidsx, centroidsy, color='black')
    plt.show()


main()
