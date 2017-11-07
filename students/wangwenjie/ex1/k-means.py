#coding=utf-8
import random
import math
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt


class kmeans():

	def set_config(self,label_num=7,file_in="synthetic_data/Aggregation.txt",file_out="synthetic_data/Aggregation_res.txt"):
		

		self.file_in = file_in
		self.file_out = file_out
		self.label_num = label_num

		# self.data_loc = np.zeros([self.maxn, 2], dtype=np.float32) 
		# self.data_label = np.zeros([self.maxn], dtype=np.int32)
		# self.data_ground_truth = np.zeros([self.maxn], dtype=np.int32)
		self.data_loc=[]
		self.data_label=[]
		self.data_ground_truth=[]
		self.data_num = 0
		self.data_center =  np.zeros([self.label_num,2], dtype=np.float32) 
		self.error_ = 0.01
		self.final_accuary = 0

	def load_data(self):
		
		file = open(self.file_in,"r")
		cou = 0
		for line in file:
			res = line.split(",")
			self.data_loc.append([])
			self.data_loc[cou].append(float(res[0]))
			self.data_loc[cou].append(float(res[1]))
			self.data_label.append(0)
			self.data_ground_truth.append(int(res[2]))
			cou += 1
		self.data_loc = np.array(self.data_loc).astype(np.float32)
		self.data_label = np.array(self.data_label).astype(np.int32)
		self.data_ground_truth = np.array(self.data_ground_truth).astype(np.int32)
		self.data_num = cou 

	def calc_distance(self,x1,y1,x2,y2):
		
		return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))


	def center_init(self):
		
		first = random.randint(0,self.data_num-1)

		self.data_center[0,:] = self.data_loc[first,:]

		data_distance = np.zeros([self.data_num], dtype=np.float32) 
		sum = 0
		for i in range(self.data_num):
			data_distance[i] = self.calc_distance(
				self.data_loc[i,0],self.data_loc[i,1],self.data_loc[first,0],self.data_loc[first,1])
			sum += data_distance[i]

		if self.label_num < 1 :
			print "label_num < 1 "

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
			distance = self.calc_distance(x,y,self.data_center[i,0],self.data_center[i,1])
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

	def calc_accuary(self):
		
		self.fianl_accuary = adjusted_mutual_info_score(self.data_label, self.data_ground_truth)
		print "{} all : {} , accuary : {} \n".format(self.file_in,self.data_num, self.fianl_accuary)

	def data_output(self):
		
		file = open(self.file_out,"w")
		for i in range(self.data_num):
			file.write("{},{},{}\n".format(self.data_loc[i,0],self.data_loc[i,1],self.data_label[i]))


if __name__ == "__main__":
	
	# k = kmeans()
	# k.set_config()
	# k.load_data()
	# k.center_init()
	# k.center_update()
	# k.calc_accuary()
	# k.data_output()



	k = kmeans()
	k.set_config(2,"flame.txt","synthetic_data/flame_res.txt")
	k.load_data()
	k.center_init()
	k.center_update()
	k.calc_accuary()
	k.data_output()
	
	fig = plt.figure(figsize=(10, 20))
	## visualization
	img0 = fig.add_subplot(421)
	img0.set_title("orignal")
	plt.scatter(k.data_loc[:, 0], k.data_loc[:, 1],c=k.data_ground_truth)
	img1 = fig.add_subplot(422)
	plt.scatter(k.data_loc[:, 0], k.data_loc[:, 1],c=k.data_label)
	img1.set_title("k-means")

	k = kmeans()
	k.set_config(15,"R15.txt","synthetic_data/R15_res.txt")
	k.load_data()
	k.center_init()
	k.center_update()
	k.calc_accuary()
	k.data_output()

	img0 = fig.add_subplot(423)
	img0.set_title("orignal")
	plt.scatter(k.data_loc[:, 0], k.data_loc[:, 1],c=k.data_ground_truth)
	img1 = fig.add_subplot(424)
	plt.scatter(k.data_loc[:, 0], k.data_loc[:, 1],c=k.data_label)
	img1.set_title("k-means")

	

	k = kmeans()
	k.set_config(7,"Aggregation.txt","synthetic_data/Aggregation_res.txt")
	k.load_data()
	k.center_init()
	k.center_update()
	k.calc_accuary()
	k.data_output()

	img0 = fig.add_subplot(425)
	img0.set_title("orignal")
	plt.scatter(k.data_loc[:, 0], k.data_loc[:, 1],c=k.data_ground_truth)
	img1 = fig.add_subplot(426)
	plt.scatter(k.data_loc[:, 0], k.data_loc[:, 1],c=k.data_label)
	img1.set_title("k-means")

	k = kmeans()
	k.set_config(24,"mix.txt","synthetic_data/mix_res.txt")
	k.load_data()
	k.center_init()
	k.center_update()
	k.calc_accuary()
	k.data_output()

	
	img0 = fig.add_subplot(427)
	img0.set_title("orignal")
	plt.scatter(k.data_loc[:, 0], k.data_loc[:, 1],c=k.data_ground_truth)
	img1 = fig.add_subplot(428)
	plt.scatter(k.data_loc[:, 0], k.data_loc[:, 1],c=k.data_label)
	img1.set_title("k-means")

	plt.show()

	


