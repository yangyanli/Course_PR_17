'''as we all know, every image is 28*28 we calculate the sum of pixel
value in each column, through this step, we can get a vector(28*1), then
we cluster them by consine similarity.

'''

from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import VisualofMnist
# My libraries
import mnist_loader
from sklearn import manifold

def Getlabel(data):




    darkness = avg_darknesses(data)

    kmeans=KMeans(n_clusters=10).fit(darkness)

    return kmeans.labels_
def avg_darknesses(data):
    darknesses =[]
    for image in data:
        temp=[]
        for i in range(28):
            temp.append(np.sum(image[i:len(image):28]))
        darknesses.append(temp)

    for i in range(len(darknesses)):
        darknesses[i]=np.divide(darknesses[i],float(np.sum(darknesses[i])))

    return darknesses




if __name__ == "__main__":
    training_data, validation_data, test_data, data, label = mnist_loader.load_data_wrapper()
    label=Getlabel(data)

    data2 = []
    for d in data:
        data2.append(np.reshape(d, 784))
    data2 = np.array(data2)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    X_tsne = tsne.fit_transform(data2[0:1000])

    VisualofMnist.plot_embedding(X_tsne, label)


