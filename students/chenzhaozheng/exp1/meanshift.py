import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.datasets import *
from sklearn.metrics import adjusted_mutual_info_score

class meanshift():
	# Aggregation-7,flame-2,R15,mix-24
	file_in = 'data/synthetic_data/flame.txt'
	original_X = []
	ground_truth = []
	label = []
	n_iterations = 5
	look_distance = 3
	bandwidth = 2
	same_distance = 1
	n = 0
	def load_data(self):
		print "loading data..."
		data_loc = []
		file = open(self.file_in,"r")
		num = 0
		for line in file:
			# print line
			res = line.split(",")
			self.original_X.append([])
			self.original_X[num].append(float(res[0]))
			self.original_X[num].append(float(res[1]))
			self.ground_truth.append(int(res[2]))
			num += 1 
		self.original_X = np.array(self.original_X)
		self.n = num
		print "There is %d points in dataset" % (self.n)
	def euclid_distance(self,x1,x2):
		return np.sqrt(np.sum(x1-x2)**2)

	def neighbourhood_points(self,X,x_centroid):
		# eligible_X = []
		# for x in X:
		# 	dis = self.euclid_distance(x,x_centroid)
		# 	if(dis<=self.look_distance):
		# 		eligible_X.append(x)
		return X

	def gaussian_kernel(self,distance):
		val = (1/(self.bandwidth*math.sqrt(2*math.pi)))*np.exp(-0.5*(distance/self.bandwidth)**2)
		return val

	def calc_accuracy(self):
		accuary = adjusted_mutual_info_score(self.label,self.ground_truth)
		print "accuary : {} \n".format(accuary)
	def update(self):
		X = np.copy(self.original_X)
		past_X = []
		print "iterating..."
		for it in range(self.n_iterations):
			for i,x in enumerate(X):# O(n)
				neighbours = self.neighbourhood_points(X,x) #O(n)
				numerator = 0
				denominator = 0
				for neighbour in neighbours:
					dis = self.euclid_distance(neighbour,x)
					weight = self.gaussian_kernel(dis)
					numerator += (weight*neighbour)
					denominator += weight
				new_x = numerator/denominator
				X[i] = new_x
			past_X.append(np.copy(X))
		
		self.label = np.zeros([self.n], dtype=np.int32) 
		self.label = self.label.tolist()
		num = 0
		for i,x in enumerate(X):
			for j in range(i):
				if self.euclid_distance(X[i],X[j])< self.same_distance:
					self.label[i] = self.label[j] 
					break;
		 	if(self.label[i]==0):
		 		num += 1
		 		self.label[i] = num
		print "The points in the data set are divided into %d categories"%(num)
	def plot(self):
		print "ploting..."
		figure = fig = plt.figure(figsize=(30, 5))
		plt.subplot(121)
		plt.title('Initial state')
		plt.scatter(self.original_X[:,0], self.original_X[:,1], c=self.ground_truth)	
		plt.subplot(122)		
		plt.title('mean-shift')
		plt.scatter(self.original_X[:,0], self.original_X[:,1], c=self.label)
		plt.show()
if __name__ == '__main__':
	m = meanshift()
	m.load_data()
	m.update()
	m.calc_accuracy()
	m.plot()
