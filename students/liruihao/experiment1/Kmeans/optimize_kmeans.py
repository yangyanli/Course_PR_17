from matplotlib.pyplot import *
import pandas as pd
import random
import numpy as np
from collections import defaultdict


# function to calculate distance
def distance(p1, p2):
	return ((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** (0.5)


# randomly generate around 100 cartesian coordinates
all_points = []

read_data = pd.read_csv('flame.csv')
# read_data = pd.read_csv('Aggregation.csv')
# read_data = pd.read_csv('R15.csv')
read_data = read_data.values
size = len(read_data)

for i in range(0, size):
	data = [i, read_data[i][0], read_data[i][1], 0]
	all_points.append(data)

centerPoint1Index = random.randint(0, size - 1)
centerPoint2Index = random.randint(0, size - 1)

while centerPoint2Index == centerPoint1Index:
	centerPoint2Index = random.randint(0, size - 1)

centerPoint1Index_old = centerPoint1Index
centerPoint2Index_old = centerPoint2Index

centerPoint1 = all_points[centerPoint1Index]
centerPoint2 = all_points[centerPoint2Index]
centerPoint1_old = centerPoint1
centerPoint2_old = centerPoint2

for iterate in range(0, 100):
	for point in all_points:
		if (distance(point, centerPoint1) < distance(point, centerPoint2)):
			point[3] = 1
		else:
			point[3] = 2

	centerPoint1Index_old = centerPoint1Index
	centerPoint2Index_old = centerPoint2Index
	centerPoint1_old = all_points[centerPoint1Index_old]
	centerPoint2_old = all_points[centerPoint2Index_old]

	# 1
	pointIndex = []
	i = 0
	j = 0
	for i in range(0, size):
		if (all_points[i][3] == 1):
			point = all_points[i]
			distances = 0
			for j in range(i, size):
				if (all_points[i][3] == 1):
					distances += distance(all_points[i], all_points[j])
			pointIndex.append([i, distances])

	centerPoint1Index = 0
	m = pointIndex[0][1]
	for k in range(0, len(pointIndex)):
		if (pointIndex[k][1] < m):
			m = pointIndex[k][1]
			centerPoint1Index = k
	centerPoint1 = all_points[centerPoint1Index]
	print(centerPoint1)

	# 2
	pointIndex = []
	i = 0
	j = 0
	for i in range(0, size):
		if (all_points[i][3] == 2):
			point = all_points[i]
			distances = 0
			for j in range(i, size):
				if (all_points[i][3] == 2):
					distances += distance(all_points[i], all_points[j])
			pointIndex.append([i, distances])

	centerPoint2Index = 0
	m = pointIndex[0][0]
	for k in range(0, len(pointIndex)):
		if (pointIndex[k][1] < m):
			m = pointIndex[k][1]
			centerPoint2Index = k
	centerPoint2 = all_points[centerPoint2Index]
	print(centerPoint2)

	loss = distance(centerPoint1, centerPoint1_old) + distance(centerPoint2, centerPoint2_old)
	print(loss)

# after the points are asssigned correnponding labels, we group them
cluster_list = defaultdict(lambda: [[], []])
for point in all_points:
	cluster_list[point[3]][0].append(point[1])
	cluster_list[point[3]][1].append(point[2])

markers = ['+', '*', '.', 'o', 'd', '^', 'v', '>', '<', 'p', '^', 's', '1', '2', '3', '4', 'h', 'H', 'D']

# plotting the clusters
i = 0
for value in cluster_list:
	cluster = cluster_list[value]
	plot(cluster[0], cluster[1], markers[i])
	i = (i + 1) % 19

plot(centerPoint1[1], centerPoint1[2], "x")
plot(centerPoint2[1], centerPoint2[2], "x")

axis((0, 15, 10, 30)) # flame
# axis((0, 40, 0, 30))  # Aggregation
# axis((0, 20, 0, 20))  # R15
show()