#coding=utf-8
import random
import math
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt


class kmeans():
	# Aggregation-7,flame-2,R15,mix-24
	file_in = 'data/synthetic_data/mix.txt'
	file_out = "data/synthetic_data/Aggregation_res.txt"

	label_num = 24
	#data_loc = np.zeros([maxn, 2], dtype=np.float32) 
	#data_label = np.zeros([maxn], dtype=np.int32)
	#data_ground_truth = np.zeros([maxn], dtype=np.int32)
	data_center =  np.zeros([label_num,2], dtype=np.float32) 
	data_loc = []
	data_ground_truth = []
	data_label = []
	#data_center = []

	data_num = 0
	error_ = 0.05
	final_accuary = 0

	def load_data(self):
		
		file = open(self.file_in,"r")
		cur = 0
		for line in file:
			# print line
			res = line.split(",")
			self.data_loc.append([])
			self.data_loc[cur].append(float(res[0]))
			self.data_loc[cur].append(float(res[1]))
			self.data_label.append(0)
			self.data_ground_truth.append(int(res[2]))
			cur += 1
		self.data_num = cur 
		self.data_loc = np.array(self.data_loc)
	def euclid_distance(self,x1,x2):
		return np.sqrt(np.sum(x1-x2)**2)


	def center_init(self):# k-means++
		
		first = random.randint(0,self.data_num-1)

		self.data_center[0,:] = self.data_loc[first,:]

		data_distance = [] 
		sum = 0
		for i in range(self.data_num):
			data_distance.append(self.euclid_distance(self.data_loc[i,:],self.data_center[0,:]))
			sum += data_distance[i]

		for i in range(1,self.label_num,1):
			random_sum = random.uniform(0,sum)
			for j in range(self.data_num):
				random_sum -= data_distance[j]
				if random_sum <= 0:
					self.data_center[i,:] = self.data_loc[j,:]
					break

	def find_nearest_center(self,x,y):

		min_distance = float("inf")
		index = 0
		for i in range(self.label_num):
			distance = self.euclid_distance([x,y],self.data_center[i,:])
			if distance < min_distance:
				min_distance = distance
				index = i
		return index,min_distance



	def center_update(self):

		while True:
			center_sum_loc = np.zeros([self.label_num,2],dtype=np.float32)
			center_num = np.zeros([self.label_num],dtype=np.int32)
			change_num = 0
			for i in range(self.data_num):
				center,distance = self.find_nearest_center(self.data_loc[i,0],self.data_loc[i,1])
				center_sum_loc[center,:] += self.data_loc[i,:]
				center_num[center] += 1
				if(self.data_label[i]!=center):
					change_num += 1
					self.data_label[i]=center

			if change_num *1.0 / self.data_num < self.error_:
				break

			for i in range(self.label_num):
				self.data_center[i,0] = center_sum_loc[i,0]/center_num[i]
				self.data_center[i,1] = center_sum_loc[i,1]/center_num[i]

	def calc_accuracy(self):
		
		self.fianl_accuary = adjusted_mutual_info_score(self.data_label, self.data_ground_truth)
		print "all : {} , accuary : {} \n".format(self.data_num, self.fianl_accuary)

	def data_output(self):
		
		file = open(self.file_out,"w")
		for i in range(self.data_num):
			file.write("{},{},{}\n".format(self.data_loc[i,0],self.data_loc[i,1],self.data_label[i]))


if __name__ == "__main__":
	
	k = kmeans()
	k.load_data()
	k.center_init()
	k.center_update()
	k.calc_accuracy()
	#k.data_output()

	## visualization
	fig = plt.figure(figsize=(30, 5))
	img0 = fig.add_subplot(121)
	img0.set_title("orignal")
	plt.scatter(k.data_loc[:, 0], k.data_loc[:, 1],c=k.data_ground_truth)
	img1 = fig.add_subplot(122)
	plt.scatter(k.data_loc[:, 0], k.data_loc[:, 1],c=k.data_label)
	img1.set_title("k-means")

	plt.show()

	



