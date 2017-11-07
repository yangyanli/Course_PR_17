#coding:utf-8
import rcc
import struct
import numpy as np
from time import time
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import (cluster,manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

''' pca and tsne for dimesion reduction and visualization
iris = load_iris()
X_tsne = TSNE(learning_rate=100).fit_transform(iris.data)
X_pca = PCA().fit_transform(iris.data)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
'''

print"load data ..."

### flame.txt for cluster
data_flame_X = []
data_flame_Y = []

with open('flame.txt', 'r') as f:
	for line in f:
		line_split = line.strip().replace(' ','').split(',')
		x = np.array([float(s) for s in line_split[:-1]])
		y = int(line_split[-1])
		data_flame_X.append(x)
		data_flame_Y.append(y)

data_flame_X = np.array(data_flame_X).astype(np.float32)
data_flame_Y = np.array(data_flame_Y)

### R15.txt for cluster
data_R15_X = []
data_R15_Y = []

with open('R15.txt', 'r') as f:
	for line in f:
		line_split = line.strip().replace(' ','').split(',')
		x = np.array([float(s) for s in line_split[:-1]])
		y = int(line_split[-1])
		data_R15_X.append(x)
		data_R15_Y.append(y)

data_R15_X = np.array(data_R15_X).astype(np.float32)
data_R15_Y = np.array(data_R15_Y)


### mix.txt for cluster
data_mix_X = []
data_mix_Y = []

with open('mix.txt', 'r') as f:
	for line in f:
		line_split = line.strip().replace(' ','').split(',')
		x = np.array([float(s) for s in line_split[:-1]])
		y = int(line_split[-1])
		data_mix_X.append(x)
		data_mix_Y.append(y)

data_mix_X = np.array(data_mix_X).astype(np.float32)
data_mix_Y = np.array(data_mix_Y)

### pendigits.txt rcc for cluster and t-sne for visualization
data_pen_X = []
data_pen_Y = []

with open('pendigits.txt', 'r') as f:
	for line in f:
		line_split = line.strip().replace(' ','').split(',')
		x = np.array([int(s) for s in line_split[:-1]])
		y = int(line_split[-1])
		data_pen_X.append(x)
		data_pen_Y.append(y)

data_pen_X = np.array(data_pen_X).astype(np.float32)
data_pen_Y = np.array(data_pen_Y)

### mnist data_load
filename = 'train-images.idx3-ubyte'
binfile = open(filename , 'rb')
buf = binfile.read()
 
index = 0
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
index += struct.calcsize('>IIII')

numImages = 1000
data_mnist_X = np.zeros([numImages,numColumns*numRows],dtype = np.int32)

for i in range(numImages):
	
	im = struct.unpack_from('>784B' ,buf, index)
	index += struct.calcsize('>784B')
	im = np.array(im)

	data_mnist_X[i,:]=im

filename = 'train-labels.idx1-ubyte'
binfile = open(filename , 'rb')
buf = binfile.read()
 
index = 0
magic_label, numLabels  = struct.unpack_from('>II' , buf , index)
index += struct.calcsize('>II')

numLabels = numImages
data_mnist_Y = np.zeros([numLabels],dtype = np.int32)

for i in range(numLabels):
	
	im = struct.unpack_from('>B' ,buf, index)
	index += struct.calcsize('>B')
	im = np.array(im)

	data_mnist_Y[i]=im

### flame 
print"flame : begin "

flame_data_num = len(data_flame_X)
flame_data_labels = np.max(data_flame_Y)

print "		data_num : {}".format(flame_data_num)
print "		labels_num : {}".format(flame_data_labels)

##	kmeans
print "\nflame : kmeans"

temp = time()
estimator = cluster.KMeans(n_clusters=2)#构造聚类器
estimator.fit(data_flame_X)#聚类
flame_kmeans_label = estimator.labels_ #获取聚类标签
# centroids = estimator.cluster_centers_ #获取聚类中心
# inertia = estimator.inertia_ # 获取聚类准则的最后值
time_flame_kmeans = time()-temp
acc_flame_kmeans = adjusted_mutual_info_score(data_flame_Y,flame_kmeans_label)
labels_num_flame_kmeans = np.max(flame_kmeans_label)+1

print "		time : {}".format(time_flame_kmeans)
print "		AMI : {}".format(acc_flame_kmeans)
print "		labels_num_predict : {}".format(labels_num_flame_kmeans)

##	meanShift
print "\nflame : meanShift"

temp = time()
bandwidth = cluster.estimate_bandwidth(data_R15_X, quantile = 0.2, n_samples = 30)  
estimator = cluster.MeanShift(bandwidth = bandwidth ) 
estimator.fit(data_flame_X)#聚类
flame_MeanShift_label = estimator.labels_ #获取聚类标签

time_flame_meanshift = time()-temp
acc_flame_meanshift = adjusted_mutual_info_score(data_flame_Y,flame_MeanShift_label)
labels_num_flame_meanshift = np.max(flame_MeanShift_label)+1

print "		time : {}".format(time_flame_meanshift)
print "		AMI : {}".format(acc_flame_meanshift)
print "		labels_num_predict : {}".format(labels_num_flame_meanshift)


##	spectral
print "\nflame : spectral"

temp = time()

spectral = cluster.SpectralClustering(
        n_clusters=3, eigen_solver='arpack',
        affinity="nearest_neighbors")

spectral.fit(data_flame_X)#聚类
flame_spectral_label = spectral.labels_ #获取聚类标签

time_flame_spectral = time()-temp
acc_flame_spectral = adjusted_mutual_info_score(data_flame_Y,flame_spectral_label)
labels_num_flame_spectral = np.max(flame_spectral_label)+1

print "		time : {}".format(time_flame_spectral)
print "		AMI : {}".format(acc_flame_spectral)
print "		labels_num_predict : {}".format(labels_num_flame_spectral)



print "\nflame : RCC"
## 	rcc 
temp = time()
clusterer = rcc.rcc_cluster(measure='cosine')
flame_rcc_label,flame_rcc_U = clusterer.fit(data_flame_X)

time_flame_rcc = time()-temp
acc_flame_rcc = adjusted_mutual_info_score(data_flame_Y,flame_rcc_label)
labels_num_flame_rcc = np.max(flame_rcc_label)+1

print "		time : {}".format(time_flame_rcc)
print "		AMI : {}".format(acc_flame_rcc)
print "		labels_num_predict : {}".format(labels_num_flame_rcc)


## visualization
fig = plt.figure(figsize=(30, 5))
img0 = fig.add_subplot(361)
img0.set_title("orignal")
plt.scatter(data_flame_X[:, 0], data_flame_X[:, 1],c=data_flame_Y)

img1 = fig.add_subplot(362)
img1.set_title("k-means")
plt.scatter(data_flame_X[:, 0], data_flame_X[:, 1],c=flame_kmeans_label)

img2 = fig.add_subplot(363)
img2.set_title("MeanShift")
plt.scatter(data_flame_X[:, 0], data_flame_X[:, 1],c=flame_MeanShift_label)

img3 = fig.add_subplot(364)
img3.set_title("spectral")
plt.scatter(data_flame_X[:, 0], data_flame_X[:, 1],c=flame_spectral_label)

img4 = fig.add_subplot(365)
img4.set_title("RCC")
plt.scatter(data_flame_X[:, 0], data_flame_X[:, 1],c=flame_rcc_label)

img5 = fig.add_subplot(366)
img5.set_title("RCC_U")
plt.scatter(flame_rcc_U[:, 0], flame_rcc_U[:, 1],c=flame_rcc_label)

### R15 
print"\n\nR15 begin"

R15_data_num = len(data_R15_X)
R15_data_labels = np.max(data_R15_Y)

print "		data_num : {}".format(R15_data_num)
print "		labels_num : {}".format(R15_data_labels)


##	kmeans
print"\nR15 : kmeans"

temp = time()
estimator = cluster.KMeans(n_clusters=15)#构造聚类器
estimator.fit(data_R15_X)#聚类
R15_kmeans_label = estimator.labels_ #获取聚类标签
# centroids = estimator.cluster_centers_ #获取聚类中心
# inertia = estimator.inertia_ # 获取聚类准则的最后值

time_R15_kmeans = time()-temp
acc_R15_kmeans = adjusted_mutual_info_score(data_R15_Y,R15_kmeans_label)
labels_num_R15_kmeans = np.max(R15_kmeans_label)+1

print "		time : {}".format(time_R15_kmeans)
print "		AMI : {}".format(acc_R15_kmeans)
print "		labels_num_predict : {}".format(labels_num_R15_kmeans)


##	meanshift 
print"\nR15 : meanshift"
temp = time()
bandwidth = cluster.estimate_bandwidth(data_R15_X, quantile = 0.2, n_samples = 50)  
estimator = cluster.MeanShift(bandwidth = bandwidth) 
estimator.fit(data_R15_X)#聚类
R15_MeanShift_label = estimator.labels_ #获取聚类标签

time_R15_meanshift = time()-temp
acc_R15_meanshift = adjusted_mutual_info_score(data_R15_Y,R15_MeanShift_label)
labels_num_R15_meanshift = np.max(R15_MeanShift_label)+1


print "		time : {}".format(time_R15_meanshift)
print "		AMI : {}".format(acc_R15_meanshift)
print "		labels_num_predict : {}".format(labels_num_R15_meanshift)

##	spectral
print "\nR15 : spectral"

temp = time()

spectral = cluster.SpectralClustering(
        n_clusters=3, eigen_solver='arpack',
        affinity="nearest_neighbors")

spectral.fit(data_R15_X)#聚类
R15_spectral_label = spectral.labels_ #获取聚类标签

time_R15_spectral = time()-temp
acc_R15_spectral = adjusted_mutual_info_score(data_R15_Y,R15_spectral_label)
labels_num_R15_spectral = np.max(R15_spectral_label)+1

print "		time : {}".format(time_R15_spectral)
print "		AMI : {}".format(acc_R15_spectral)
print "		labels_num_predict : {}".format(labels_num_R15_spectral)



##	RCC
print"\nR15 : RCC"

temp = time()
clusterer = rcc.rcc_cluster(measure='cosine')
R15_rcc_label,R15_rcc_U = clusterer.fit(data_R15_X)
time_R15_RCC = time()-temp
acc_R15_RCC = adjusted_mutual_info_score(data_R15_Y,R15_rcc_label)
labels_num_R15_RCC = np.max(R15_rcc_label)+1

print "		time : {}".format(time_R15_RCC)
print "		AMI : {}".format(acc_R15_RCC)
print "		labels_num_predict : {}".format(labels_num_R15_RCC)


fig.add_subplot(367)
plt.scatter(data_R15_X[:, 0], data_R15_X[:, 1],c=data_R15_Y)
fig.add_subplot(368)
plt.scatter(data_R15_X[:, 0], data_R15_X[:, 1],c=R15_kmeans_label)
fig.add_subplot(3,6,9)
plt.scatter(data_R15_X[:, 0], data_R15_X[:, 1],c=R15_MeanShift_label)
fig.add_subplot(3,6,10)
plt.scatter(data_R15_X[:, 0], data_R15_X[:, 1],c=R15_spectral_label)
fig.add_subplot(3,6,11)
plt.scatter(data_R15_X[:, 0], data_R15_X[:, 1],c=R15_rcc_label)
fig.add_subplot(3,6,12)
plt.scatter(R15_rcc_U[:, 0], R15_rcc_U[:, 1],c=R15_rcc_label)

### mix
print "\n\nmix : begin"

mix_data_num = len(data_mix_X)
mix_data_labels = np.max(data_mix_Y)

print "		data_num : {}".format(mix_data_num)
print "		labels_num : {}".format(mix_data_labels)


##	kmeans
print "\nmix : kmeans"

temp = time()
estimator = cluster.KMeans(n_clusters=np.max(data_mix_Y))#构造聚类器
estimator.fit(data_mix_X)#聚类
mix_kmeans_label = estimator.labels_ #获取聚类标签
# centroids = estimator.cluster_centers_ #获取聚类中心
# inertia = estimator.inertia_ # 获取聚类准则的最后值

time_mix_kmeans = time()-temp
acc_mix_kmeans = adjusted_mutual_info_score(data_mix_Y,mix_kmeans_label)
labels_num_mix_kmeans = np.max(mix_kmeans_label)+1


print "		time : {}".format(time_mix_kmeans)
print "		AMI : {}".format(acc_mix_kmeans)
print "		labels_num_predict : {}".format(labels_num_mix_kmeans)


##	meanshift 
print "\nmix : meanshift"

temp = time()
bandwidth = cluster.estimate_bandwidth(data_mix_X, quantile = 0.15, n_samples = 500)  
estimator = cluster.MeanShift(bandwidth = bandwidth ) 
estimator.fit(data_mix_X)#聚类
mix_MeanShift_label = estimator.labels_ #获取聚类标签

time_mix_meanshift = time()-temp
acc_mix_meanshift = adjusted_mutual_info_score(data_mix_Y,mix_MeanShift_label)
labels_num_mix_meanshift = np.max(mix_MeanShift_label)+1

print "		time : {}".format(time_mix_meanshift)
print "		AMI : {}".format(acc_mix_meanshift)
print "		labels_num_predict : {}".format(labels_num_mix_meanshift)

##	spectral
print "\nmix : spectral"

temp = time()

spectral = cluster.SpectralClustering(
        n_clusters=3, eigen_solver='arpack',
        affinity="nearest_neighbors")

spectral.fit(data_mix_X)#聚类
mix_spectral_label = spectral.labels_ #获取聚类标签

time_mix_spectral = time()-temp
acc_mix_spectral = adjusted_mutual_info_score(data_mix_Y,mix_spectral_label)
labels_num_mix_spectral = np.max(mix_spectral_label)+1

print "		time : {}".format(time_R15_spectral)
print "		AMI : {}".format(acc_mix_spectral)
print "		labels_num_predict : {}".format(labels_num_mix_spectral)


##	RCC
print "\nmix : RCC"

temp = time()
clusterer = rcc.rcc_cluster(measure='cosine')
mix_rcc_label,mix_rcc_U = clusterer.fit(data_mix_X)

time_mix_RCC = time()-temp
acc_mix_RCC = adjusted_mutual_info_score(data_mix_Y,mix_rcc_label)
labels_num_mix_RCC = np.max(mix_rcc_label)+1

print "		time : {}".format(time_mix_RCC)
print "		AMI : {}".format(acc_mix_RCC)
print "		labels_num_predict : {}".format(labels_num_mix_RCC)



fig.add_subplot(3,6,13)
plt.scatter(data_mix_X[:, 0], data_mix_X[:, 1],c=data_mix_Y)
fig.add_subplot(3,6,14)
plt.scatter(data_mix_X[:, 0], data_mix_X[:, 1],c=mix_kmeans_label)
fig.add_subplot(3,6,15)
plt.scatter(data_mix_X[:, 0], data_mix_X[:, 1],c=mix_MeanShift_label)
fig.add_subplot(3,6,16)
plt.scatter(data_mix_X[:, 0], data_mix_X[:, 1],c=mix_spectral_label)
fig.add_subplot(3,6,17)
plt.scatter(data_mix_X[:, 0], data_mix_X[:, 1],c=mix_rcc_label)
fig.add_subplot(3,6,18)
plt.scatter(mix_rcc_U[:, 0], mix_rcc_U[:, 1],c=mix_rcc_label)


## pendigits.txt rcc for cluster and t-sne for visualization

print "\n\npen digits : begin "

pen_data_num = len(data_pen_X)
pen_data_labels = np.max(data_pen_Y)+1

print "		data_num : {}".format(pen_data_num)
print "		labels_num : {}".format(pen_data_labels)

##	kmeans
print "\npen digits : kmeans "

temp = time()
estimator = cluster.KMeans(n_clusters=10)#构造聚类器
estimator.fit(data_pen_X)#聚类
pen_kmeans_label = estimator.labels_ #获取聚类标签
# centroids = estimator.cluster_centers_ #获取聚类中心
# inertia = estimator.inertia_ # 获取聚类准则的最后值

time_pen_kmeans = time()-temp
acc_pen_kmeans = adjusted_mutual_info_score(data_pen_Y,pen_kmeans_label)
labels_num_pen_kmeans = np.max(pen_kmeans_label)+1

print "		time : {}".format(time_pen_kmeans)
print "		AMI : {}".format(acc_pen_kmeans)
print "		labels_num_predict : {}".format(labels_num_pen_kmeans)

##	meanshift 
print "\npen digits : meanshift "

temp = time()

bandwidth = cluster.estimate_bandwidth(data_pen_X, quantile = 0.1, n_samples = 20)  
estimator = cluster.MeanShift(bandwidth = bandwidth) 
estimator.fit(data_pen_X)#聚类
pen_MeanShift_label = estimator.labels_ #获取聚类标签

time_pen_meanshift = time()-temp
acc_pen_meanshift = adjusted_mutual_info_score(data_pen_Y,pen_MeanShift_label)
labels_num_pen_meanshift = np.max(pen_MeanShift_label)+1

print "		time : {}".format(time_pen_meanshift)
print "		AMI : {}".format(acc_pen_meanshift)
print "		labels_num_predict : {}".format(labels_num_pen_meanshift)

##	spectral
print "\npen digits: spectral"

temp = time()

spectral = cluster.SpectralClustering(
        n_clusters=3, eigen_solver='arpack',
        affinity="nearest_neighbors")

spectral.fit(data_pen_X)#聚类
pen_spectral_label = spectral.labels_ #获取聚类标签

time_pen_spectral = time()-temp
acc_pen_spectral = adjusted_mutual_info_score(data_pen_Y,pen_spectral_label)
labels_num_pen_spectral = np.max(pen_spectral_label)+1

print "		time : {}".format(time_pen_spectral)
print "		AMI : {}".format(acc_pen_spectral)
print "		labels_num_predict : {}".format(labels_num_pen_spectral)


print "\npen digits : RCC "

temp = time()
clusterer = rcc.rcc_cluster(measure='cosine')
P,U = clusterer.fit(data_pen_X)

time_pen_RCC = time()-temp
acc_pen_RCC = adjusted_mutual_info_score(data_pen_Y,P)
labels_num_pen_RCC = np.max(P)+1

print "		time : {}".format(time_pen_RCC)
print "		AMI : {}".format(acc_pen_RCC)
print "		labels_num_predict : {}".format(labels_num_pen_RCC)



print("\n\nComputing t-SNE embedding for pendigits")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
U_tsne = tsne.fit_transform(U)
X_tsne = tsne.fit_transform(data_pen_X)

print("Done  : Computing t-SNE embedding " )


plt.figure(figsize=(30, 5))
plt.subplot(161)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=data_pen_Y)
plt.subplot(162)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=pen_kmeans_label)
plt.subplot(163)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=pen_MeanShift_label)
plt.subplot(164)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=pen_spectral_label)
plt.subplot(165)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=P)
plt.subplot(166)
plt.scatter(U_tsne[:, 0], U_tsne[:, 1],c=P)

 



### mnist rcc for cluster and t-sne for visualization

# print "\n\nmnist : begin"

# mnist_data_num = len(data_mnist_X)
# mnist_data_labels = np.max(data_mnist_Y)+1

# print "		data_num : {}".format(mnist_data_num)
# print "		labels_num : {}".format(mnist_data_labels)


# ##	kmeans
# print("\nmnist : kmeans")

# temp = time()
# estimator = cluster.KMeans(n_clusters=10)#构造聚类器
# estimator.fit(data_mnist_X)#聚类
# mnist_kmeans_label = estimator.labels_ #获取聚类标签
# # centroids = estimator.cluster_centers_ #获取聚类中心
# # inertia = estimator.inertia_ # 获取聚类准则的最后值

# time_mnist_kmeans = time()-temp
# acc_mnist_kmeans = adjusted_mutual_info_score(data_mnist_Y,mnist_kmeans_label)
# labels_num_mnist_kmeans = np.max(mnist_kmeans_label)+1

# print "		time : {}".format(time_mnist_kmeans)
# print "		AMI : {}".format(acc_mnist_kmeans)
# print "		labels_num_predict : {}".format(labels_num_mnist_kmeans)



# ##	meanshift 
# print("\nmnist : meanshift")
# temp = time()

# bandwidth = cluster.estimate_bandwidth(data_pen_X, quantile = 0.1, n_samples = 20)  
# estimator = cluster.MeanShift(bandwidth = bandwidth) 
# estimator.fit(data_mnist_X)#聚类
# mnist_MeanShift_label = estimator.labels_ #获取聚类标签

# time_mnist_meanshift = time()-temp
# acc_mnist_meanshift = adjusted_mutual_info_score(data_mnist_Y,mnist_MeanShift_label)
# labels_num_mnist_meanshift = np.max(mnist_MeanShift_label)+1

# print "		time : {}".format(time_mnist_meanshift)
# print "		AMI : {}".format(acc_mnist_meanshift)
# print "		labels_num_predict : {}".format(labels_num_mnist_meanshift)


# print("\nmnist : RCC")
# temp = time()

# clusterer = rcc.rcc_cluster(measure='cosine')
# mnist_P,mnist_U = clusterer.fit(data_mnist_X)

# time_mnist_RCC = time()-temp
# acc_mnist_RCC = adjusted_mutual_info_score(data_mnist_Y,mnist_P)
# labels_num_mnist_RCC = np.max(mnist_P)+1

# print "		time : {}".format(time_mnist_RCC)
# print "		AMI : {}".format(acc_mnist_RCC)
# print "		labels_num_predict : {}".format(labels_num_mnist_RCC)


# print("\n\nComputing t-SNE embedding for mnist")
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# mnist_U_tsne = tsne.fit_transform(mnist_U)
# mnist_X_tsne = tsne.fit_transform(data_mnist_X)
# print("Done  : Computing t-SNE embedding " )



# plt.subplot(256)
# plt.scatter(mnist_X_tsne[:, 0], mnist_X_tsne[:, 1],c=data_mnist_Y)
# plt.subplot(257)
# plt.scatter(mnist_X_tsne[:, 0], mnist_X_tsne[:, 1],c=mnist_kmeans_label)
# plt.subplot(258)
# plt.scatter(mnist_X_tsne[:, 0], mnist_X_tsne[:, 1],c=mnist_MeanShift_label)
# plt.subplot(259)
# plt.scatter(mnist_X_tsne[:, 0], mnist_X_tsne[:, 1],c=mnist_P)
# plt.subplot(2,5,10)
# plt.scatter(mnist_U_tsne[:, 0], mnist_U_tsne[:, 1],c=mnist_P)
 

plt.show()







