from data_util import DataUtils
import datetime  
from numpy import *
import operator

def knn(test_image, train_image, labels, k):  
    different_image = tile(test_image, (shape(train_image)[0], 1)) - train_image   
    different_image_square = different_image ** 2  
    distance_square = different_image_square.sum(axis=1)  
    distances = distance_square ** 0.5  
    sorted_distance_index = distances.argsort()  
    class_count = {}  
    for i in range(k):  
        target_label = labels[sorted_distance_index[i]]  
        class_count[target_label] = class_count.get(target_label,0) + 1  
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1),reverse=True)  
    return sorted_class_count[0][0]  
      
def recognize(train_image, train_labels, test_image, test_labels, K):  
    correct_count = 0  
    for i in range(len(test_image)):  
        if i % 100 == 0:  
            print ('total test '+str(i)+' accuracy: '+ str(float(correct_count)/(i+1)))  
        label = test_labels[i]  
        predict_label = knn(test_image[i], train_image, train_labels, K)  
        if label == predict_label:  
            correct_count += 1  
    return float(correct_count)/len(test_image)

def main():
    trainfile_X = '/Users/Lagrant/Desktop/tensorflow/dataset/train-images-idx3-ubyte'
    trainfile_y = '/Users/Lagrant/Desktop/tensorflow/dataset/train-labels-idx1-ubyte'
    testfile_X = '/Users/Lagrant/Desktop/tensorflow/dataset/t10k-images-idx3-ubyte'
    testfile_y = '/Users/Lagrant/Desktop/tensorflow/dataset/t10k-labels-idx1-ubyte'
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()

    return train_X, train_y, test_X, test_y 


def testKNN():
    train_X, train_y, test_X, test_y = main()
    K = 10  
    accuracy = recognize(train_X, train_y, test_X, test_y, K)  
    print('the total accuracy is: '+str(accuracy))

if __name__ == "__main__":
    testKNN()
