from matplotlib.pyplot import *
import pandas as pd
from collections import defaultdict


# function to calculate distance
def distance(p1, p2):
	return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (0.5)


# randomly generate around 100 cartesian coordinates
all_points = []
all_points_2 = []

# read_data = pd.read_csv('flame.csv')
# read_data = pd.read_csv('Aggregation.csv')
# read_data = pd.read_csv('R15.csv')
read_data = pd.read_csv('mix.csv')
read_data = read_data.values
size = len(read_data)

for i in range(0, size):
	data = [read_data[i][0], read_data[i][1]]
	if not data in all_points and (data[0] > 20 or data[1] < 17.2):
		all_points.append(data)
	else:
		all_points_2.append(data)

# E = 1.5  # flame
# minPts = 10  # flame

# E = 2 # Aggregation
# minPts = 15  # Aggregation

# E = 0.8  # R15
# minPts = 35  # R15

E = 0.8  # mix
minPts = 35  # mix

# find out the core points
other_points = []
core_points = []
plotted_points = []
for point in all_points:
	point.append(0)  # assign initial level 0
	total = 0
	for otherPoint in all_points:
		distance_ = distance(otherPoint, point)
		if distance_ <= E:
			total += 1

	if total > minPts:
		core_points.append(point)
		plotted_points.append(point)
	else:
		other_points.append(point)

# find border points
border_points = []
for core in core_points:
	for other in other_points:
		if distance(core, other) <= E:
			border_points.append(other)
			plotted_points.append(other)

# implement the algorithm
cluster_label = 0

for point in core_points:
	if point[2] == 0:
		cluster_label += 1
		point[2] = cluster_label

	for point2 in plotted_points:
		distance_ = distance(point2, point)
		if point2[2] == 0 and distance_ <= E:
			point2[2] = point[2]

noise_points = []
for point in all_points:
	if not point in core_points and not point in border_points:
		noise_points.append(point)


E = 1.6003  # Aggregation
minPts = 10  # Aggregation

all_points_1 = []
all_points_1 += noise_points
other_points_1 = []
core_points_1 = []
plotted_points_1 = []

for point in all_points_1:
	total = 0
	for otherPoint in all_points_1:
		distance_ = distance(otherPoint, point)
		if distance_ <= E:
			total += 1

	if total > minPts:
		core_points_1.append(point)
		plotted_points_1.append(point)
	else:
		other_points_1.append(point)

# find border points
border_points_1 = []
for core in core_points_1:
	for other in other_points_1:
		if distance(core, other) <= E:
			border_points_1.append(other)
			plotted_points_1.append(other)

# implement the algorithm

for point in core_points_1:
	if point[2] == 0:
		cluster_label += 1
		point[2] = cluster_label

	for point2 in plotted_points_1:
		distance_ = distance(point2, point)
		if point2[2] == 0 and distance_ <= E:
			point2[2] = point[2]

E = 1.5  # flame
minPts = 10  # flame

other_points_2 = []
core_points_2 = []
plotted_points_2 = []

for point in all_points_2:
	point.append(0)  # assign initial level 0
	total = 0
	for otherPoint in all_points_2:
		distance_ = distance(otherPoint, point)
		if distance_ <= E:
			total += 1

	if total > minPts:
		core_points_2.append(point)
		plotted_points_2.append(point)
	else:
		other_points_2.append(point)

# find border points
border_points_2 = []
for core in core_points_2:
	for other in other_points_2:
		if distance(core, other) <= E:
			border_points_2.append(other)
			plotted_points_2.append(other)

# implement the algorithm

for point in core_points_2:
	if point[2] == 0:
		cluster_label += 1
		point[2] = cluster_label

	for point2 in plotted_points_2:
		distance_ = distance(point2, point)
		if point2[2] == 0 and distance_ <= E:
			point2[2] = point[2]


# after the points are asssigned correnponding labels, we group them
cluster_list = defaultdict(lambda: [[], []])
for point in plotted_points:
	cluster_list[point[2]][0].append(point[0])
	cluster_list[point[2]][1].append(point[1])

for point in plotted_points_1:
	cluster_list[point[2]][0].append(point[0])
	cluster_list[point[2]][1].append(point[1])

for point in plotted_points_2:
	cluster_list[point[2]][0].append(point[0])
	cluster_list[point[2]][1].append(point[1])

markers = ['+', '*', '.', 'o', 'd', '^', 'v', '>', '<', 'p', '^', 's', '1', '2', '3', '4', 'h', 'H', 'D']

# plotting the clusters
i = 0
for value in cluster_list:
	cluster = cluster_list[value]
	plot(cluster[0], cluster[1], markers[i])
	i = (i + 1) % 19

# plot the noise points as well
noise_points = []
for point in all_points:
	if not point in core_points and not point in border_points and not point in core_points_1 and not point in border_points_1:
		noise_points.append(point)
noisex = []
noisey = []
for point in noise_points:
	noisex.append(point[0])
	noisey.append(point[1])
plot(noisex, noisey, "x")

title(str(len(cluster_list)) + " clusters created with E =" + str(E) + " Min Points=" + str(
		minPts) + " total points=" + str(len(all_points)) + " noise Points = " + str(len(noise_points)))
axis((0, 60, 0, 35))  # mix
show()
