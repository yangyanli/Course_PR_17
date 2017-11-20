import numpy as np
import mnist_loader
import matplotlib.pyplot as plt
from time import time
from sklearn import manifold


def plot_embedding(X,label):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    color = ['black', 'gray', 'red', 'darkgreen', 'blue', 'm', 'crimson', 'lime', 'coral', 'snow']
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(label[i]),
                 color=color[label[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])

    plt.show()




'''
 This is an example how to use this function.
 1. You need to load the mnist data use the library mnist_loader
 2. you need to transform the data(list) to data2(ndarray)
 3. call plot_embedding()
 notice: You'd better run less than 2000 picture once because your memory may not enough!!
'''



training_data, validation_data, test_data,data,label = mnist_loader.load_data_wrapper()
data2=[]
for d in data:
    data2.append(np.reshape(d,784))
data2=np.array(data2)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(data2[0:1000])

plot_embedding(X_tsne,label)

plt.show()

