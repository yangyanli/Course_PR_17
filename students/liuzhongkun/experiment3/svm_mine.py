from sklearn import svm
import numpy as np
import struct
from sklearn.decomposition import PCA
import math
import random

def loadImageSet(filename):
    print ("load image set", filename)
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    print ("head,", head)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    print ("load imgs finished")
    return imgs

def loadLabelSet(filename):
    print("load label set", filename)
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    print("head,", head)
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    print('load label finished')
    return labels

def K(input1,input2):
    return [K_one(i, input2) for i in input1]
def K_one(input1,input2):
    theta = 3
    tmp = -np.sum((input1 - input2) ** 2) / (2 * theta*theta)
    return math.exp(tmp)


def my_svm(data, label, test_data):
    data = np.array(data)
    len_data = len(label)
    len_w = len(data[0])
    alpha = np.zeros(len_data)
    max_iter = 100
    b = 0.0
    C = 1
    ep = 1e-3
    iter = 0
    while iter < max_iter:
        alpha_change = 0
        for i in range(len_data):
            a_l = np.multiply(alpha, label)
            pre_Li = np.sum(np.multiply(a_l,K(data,data[i,:]))) + b
            Ei = pre_Li - label[i]
            if (alpha[i] < C and (label[i]*Ei) <= -ep) or \
                    (alpha[i] >= 0 and (label[i] * Ei) > ep):
                #根据ei和ej求j

                # j = i
                # update_num = 5
                # while j == i and update_num > 0:
                #     random.seed(update_num)
                #     j = random.randint(0,len_data)
                #     update_num -= 1
                e_sub = 0
                j_max = i
                for j in range(0, len_data):
                    pre_Lj = np.sum(np.multiply(a_l, K(data,data[j]))) + b
                    Ej = pre_Lj - label[j]
                    if abs(Ei - Ej) > e_sub:
                        e_sub = abs(Ei - Ej)
                        j_max = j
                j = j_max
                if j == i:
                    continue
                # else:
                #     pre_Lj = np.sum(np.multiply(a_l, K(data, data[j]))) + b
                #     Ej = pre_Lj - label[j]
                if label[i] != label[j]:
                    L = max(0,alpha[j] - alpha[i])
                    H = min(C,C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H:
                    continue
                eta = np.sum(2 * K_one(data[i],data[j]) - K_one(data[i],data[i]) - \
                             K_one(data[j],data[j]))
                if eta == 0:
                    continue
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                alpha[j] = alpha[j] - label[j] * (Ei - Ej)/eta
                if alpha[j] > H:
                    alpha[j] = H
                elif alpha[j] < L:
                    alpha[j] = L
                if abs(alpha[j] - alpha_j_old) < 1e-3:
                    continue
                alpha[i] = alpha[i] + label[i]*label[j]*(alpha_j_old - alpha[j])
                b1 = b - Ei - label[i] * (alpha[i] - alpha_i_old) * K_one(data[i],data[i])- \
                     label[j] * (alpha[j] - alpha_j_old) * K_one(data[i], data[j])
                b2 = b - Ej - label[i] * (alpha[i] - alpha_i_old) * K_one(data[i],data[j])- \
                     label[j] * (alpha[j] - alpha_j_old) * K_one(data[j], data[j])
                if alpha[i] > 0 and alpha[i] < C:
                    b = b1
                elif alpha[j] > 0 and alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2
                alpha_change += 1
        if alpha_change > 0:
            iter = iter + 1
        else:
            break
        print('iter: ' + str(iter))
    index_list = []
    for i in range(len(alpha)):
        if abs(alpha[i]) > 1e-3:
            index_list.append(i)
    predict = []
    print(index_list)
    print(list(alpha))
    for i in range(len(test_data)):
        tmp = 0
        for j in range(len(index_list)):
            tmp += alpha[index_list[j]] * label[index_list[j]] * K_one(test_data[i],data[index_list[j]])
        if tmp + b > 0:
            pre = 1
        else :
            pre = -1
        predict.append(pre)

    return predict

def onevsrest(imgs, labels, imgs_test):
    pass
    svms = []
    for current_cluster in range(0,10):
        new_labels = labels[:]
        for i in range(len(new_labels)):
            if new_labels[i] == current_cluster:
                new_labels[i] = 1
            else:
                new_labels[i] = -1


if __name__ == '__main__':
    path = './MNIST/'
    imgs = loadImageSet(path + 'train-images.idx3-ubyte')
    labels = loadLabelSet(path + 'train-labels.idx1-ubyte')
    imgs_test = loadImageSet(path + 't10k-images.idx3-ubyte')
    labels_test = loadLabelSet(path + 't10k-labels.idx1-ubyte')

    train_num = 200
    test_num = 100

    imgs = np.array(imgs[0:train_num])
    labels = np.array(labels[0:train_num])
    imgs = imgs.reshape([-1, len(imgs[0][0])])
    labels = labels.reshape([-1])
    imgs_test = np.array(imgs_test[0:test_num])
    labels_test = np.array(labels_test[0:test_num])
    imgs_test = imgs_test.reshape([-1, len(imgs_test[0][0])])
    labels_test = labels_test.reshape([-1])
    len_train = len(imgs)
    imgs_all = np.array(list(imgs) + list(imgs_test))
    pca = PCA(n_components=100)
    pca.fit(imgs_all)
    imgs = imgs/255.0
    imgs_test = imgs_test/255.0
    imgs_ = pca.transform(imgs)
    imgs_test_ = pca.transform(imgs_test)
    for i in range(len(labels)):
        if labels[i] >= 1:
            labels[i] = 1
        else:
            labels[i] = -1
    for i in range(len(labels_test)):
        if labels_test[i] >= 1:
            labels_test[i] = 1
        else:
            labels_test[i] = -1

    result = my_svm(imgs_, labels, imgs_test_)
    print(result)
    print(list(labels_test))
    print(np.mean(result == labels_test))