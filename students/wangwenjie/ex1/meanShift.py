#coding:UTF-8

import sys
import math
import numpy as np
import matplotlib.pyplot as plt


class meanShift():


	def __init__(self,MIN_DISTANCE=0.001):

		self.MIN_DISTANCE = MIN_DISTANCE
		self.data = []
		self.label = []

	def load_data(self,path, feature_num=2):

		file = open(path)
		for line in file.readlines():
			lines = line.strip().split(",")
			tmp = []
			if len(lines) != feature_num+1:
				continue
			for i in xrange(feature_num):
				tmp.append(float(lines[i]))
			self.label.append(int(lines[feature_num]))
			self.data.append(tmp)
		file.close()
		return self.data

	def gaussian_kernel(self,distance, bandwidth=2):

		length = np.shape(distance)[0]
		res = np.mat(np.zeros((length, 1)))
		for i in xrange(length):
			res[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
			res[i, 0] = np.exp(res[i, 0])
		res = (1 / (bandwidth * math.sqrt(2 * math.pi)))*res

		return res

	def shift_point(self,point, points, kernel_bandwidth):

		points = np.mat(points)
		m,n = np.shape(points)

		point_distances = np.mat(np.zeros((m,1)))
		for i in xrange(m):
			point_distances[i, 0] = np.sqrt((point - points[i]) * (point - points[i]).T)

		point_weights = self.gaussian_kernel(point_distances, kernel_bandwidth)

		all = 0.0
		for i in xrange(m):
			all += point_weights[i, 0]

		point_shifted = point_weights.T * points / all
		return point_shifted

	def euclidean_dist(self,pointA, pointB):

		total = (pointA - pointB) * (pointA - pointB).T
		return math.sqrt(total)

	def group_points(self,mean_shift_points):

		group_assignment = []
		m,n = np.shape(mean_shift_points)
		index = 0
		index_dict = {}
		for i in xrange(m):
			item = []
			for j in xrange(n):
				item.append(str(("%5.2f" % mean_shift_points[i, j])))

			item_1 = " ".join(item)

			if item_1 not in index_dict:
				index_dict[item_1] = index
				index += 1


		for i in xrange(m):
			item = []
			for j in xrange(n):
				item.append(str(("%5.2f" % mean_shift_points[i, j])))

			item_1 = " ".join(item)
			group_assignment.append(index_dict[item_1])
		return group_assignment

	def train_mean_shift(self,points, kenel_bandwidth=2):

		mean_shift_points = np.mat(points)
		max_move = 1
		iter = 0
		m, n = np.shape(mean_shift_points)
		need_shift = [True] * m

		while max_move > self.MIN_DISTANCE:

			iter += 1
			print "iter : " + str(iter)+" max_move:"+str(max_move)
			max_move = 0
			for i in range(0, m):
				if not need_shift[i]:
					continue
				p_new = mean_shift_points[i]
				p_new_start = p_new
				p_new = self.shift_point(p_new, points, kenel_bandwidth)
				dist = self.euclidean_dist(p_new, p_new_start)

				if dist > max_move:
					max_move = dist
				if dist < self.MIN_DISTANCE:
					need_shift[i] = False

				mean_shift_points[i] = p_new
		group = self.group_points(mean_shift_points)

		return np.mat(points), mean_shift_points, group

if __name__ == "__main__":
 
	path = "flame.txt"

	m = meanShift()
	data = m.load_data(path, 2)
	_, _, cluster = m.train_mean_shift(data, 2)
	data = np.array(m.data)
 #    	## visualization
	fig = plt.figure(figsize=(10, 5))
	img0 = fig.add_subplot(121)
	img0.set_title("orignal")
	plt.scatter(data[:,0], data[:,1],c=m.label)
	img1 = fig.add_subplot(122)
	plt.scatter(data[:,0], data[:,1	],c=cluster)
	img1.set_title("meanShift")




	# path = "./synthetic_data/R15.txt"

	# m = meanShift()
	# data = m.load_data(path, 2)
	# _,_, cluster = m.train_mean_shift(data, 2)
	# data = np.array(m.data)
 # ### visualization
	# img0 = fig.add_subplot(423)
	# img0.set_title("orignal")
	# plt.scatter(data[:,0], data[:,1],c=m.label)
	# img1 = fig.add_subplot(424)
	# plt.scatter(data[:,0], data[:,1	],c=cluster)
	# img1.set_title("meanShift")


	# path = "./synthetic_data/Aggregation.txt"

	# m = meanShift()
	# data = m.load_data(path, 2)
	# _,_, cluster = m.train_mean_shift(data, 2)
	# data = np.array(m.data)
 # ### visualization
	# img0 = fig.add_subplot(425)
	# img0.set_title("orignal")
	# plt.scatter(data[:,0], data[:,1],c=m.label)
	# img1 = fig.add_subplot(426)
	# plt.scatter(data[:,0], data[:,1	],c=cluster)
	# img1.set_title("meanShift")



	# path = "./synthetic_data/mix.txt"

	# m = meanShift()
	# data = m.load_data(path, 2)
	# _,_, cluster = m.train_mean_shift(data, 2)
	# data = np.array(m.data)
 # ### visualization
	# img0 = fig.add_subplot(427)
	# img0.set_title("orignal")
	# plt.scatter(data[:,0], data[:,1],c=m.label)
	# img1 = fig.add_subplot(428)
	# plt.scatter(data[:,0], data[:,1	],c=cluster)
	# img1.set_title("meanShift")

	plt.show()
