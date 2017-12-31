# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:55:49 2017

@author: Hanlin
"""
import numpy as np
import matplotlib.pyplot as plt
import math


colors=['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']
MIN_DISTANCE = 0.0001
def gaussian_kernel(distance,bandwidth):
    m = np.shape(distance)[0]
    right = np.mat(np.zeros((m, 1)))
    for i in range(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))
    gaussian_val = left * right
    return gaussian_val

def shift_points(point,points,kernel_bandwidth):
    points = np.mat(points,dtype=float)
    m,n = np.shape(points)
    point_distances = np.mat(np.zeros((m,1)),dtype=float)
    for i in range(m):
        point_distances[i, 0] = np.sqrt((point - points[i]) * (point - points[i]).T)     
    point_weights = gaussian_kernel(point_distances, kernel_bandwidth)
    all = 0.0
    for i in range(m):
        all += point_weights[i, 0]
    point_shifted = point_weights.T * points / all
    return point_shifted

def euclidean_dist(pointA, pointB):
    total = (pointA - pointB) * (pointA - pointB).T
    return math.sqrt(total)

def distance_to_group(point,group):
    min_distance=1000000.0
    for pt in group:
        dist=euclidean_dist(point,pt)
        if dist<min_distance:
            min_distance=dist
    return min_distance

def group_points(mean_shift_points):
    group_assignment = []
    m,n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))
        item_1 = "_".join(item)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))
        item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])
    return group_assignment

def train(points,kenel_bandwidth=2):
    mean_shift_points=np.mat(data,dtype=float)
    max_min_dist=1
    iter=0
    m,n=np.shape(mean_shift_points)
    need_shift=[True]*m
    while max_min_dist>MIN_DISTANCE:
        max_min_dist=0
        iter+=1
        print("iter"+str(iter))
        for i in range(0,m):
            if not need_shift[i]:
                continue
            p_new=mean_shift_points[i];
            p_new_start=p_new
            p_new=shift_points(p_new,points, kenel_bandwidth)
            dist = euclidean_dist(p_new, p_new_start)
            if dist>max_min_dist:
                max_min_dist=dist
            if dist<MIN_DISTANCE:
                need_shift[i]=False
            mean_shift_points[i]=p_new
        print(max_min_dist,MIN_DISTANCE)
    group = group_points(mean_shift_points)
    return np.mat(points),mean_shift_points,group
          
if __name__ == '__main__':
    data=[]
    f=open("C:/Users/Hanlin/Documents/GitHub/MeanShift_py/synthetic_data/R15.txt")
    x = []
    y = []
    for line in f.readlines():  
        lines=line.split(",")
        data_temp=[]
        for i in range(2):
            data_temp.append(lines[i])
        x.append(float(lines[0]))
        y.append(float(lines[1]))
        data.append(data_temp)
    points, shift_points, cluster = train(data,0.3 )
    output = open("C:/Users/Hanlin/Documents/GitHub/MeanShift_py/R15.txt", 'w')
    
    for i in range(len(points)):
        output.write(str(points[i,0]))
        output.write(',')
        output.write(str(points[i,1]))
        output.write(',')
        output.write(str(shift_points[i,0]))
        output.write(',')
        output.write(str(shift_points[i,1]))
        output.write(',')
        output.write(str(cluster[i]))
        output.write('\n')
    output.close()