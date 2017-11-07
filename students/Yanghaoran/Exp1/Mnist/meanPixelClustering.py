'''
Firstly, You can think this is an easy classifier, And secondly, We will show how to make it an clustering algorithm!
classifier: diffenent picture has different pixels, for example  the sum pixel value of '1' is much less then '8' intuitively!
so we can classify the picture according to it's sum of pixel value.

clustering: after we calculate the sum of the pixel value, we can cluster them according the value.
'''
from sklearn import manifold
import VisualofMnist
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

# My libraries
import mnist_loader

def getlabel(data):


    darkness = avg_darknesses(data)
    darkness2=[[x] for x in darkness]
    kmeans=KMeans(n_clusters=10).fit(darkness2)
    return kmeans.labels_
def avg_darknesses(data):
    darknesses =[]
    for image in data:
       darknesses.append(np.sum(image))
   # avgs = defaultdict(float)

    return darknesses




if __name__ == "__main__":
    training_data, validation_data, test_data, data, label = mnist_loader.load_data_wrapper()
    label=getlabel(data)
    data2 = []
    for d in data:
        data2.append(np.reshape(d, 784))
    data2 = np.array(data2)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    X_tsne = tsne.fit_transform(data2[0:1000])

    VisualofMnist.plot_embedding(X_tsne, label)
