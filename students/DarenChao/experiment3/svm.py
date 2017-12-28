import numpy as np
import sklearn.datasets
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from time import time
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import pickle

class SimpleSvm:
    x = None
    y = None
    n = None # length of x or y
    kernel_type = ''
    alphas = None
    b = None
    C = None
    support_vectors = None
    tolerance = None
    max_passes = None
    gama = None

    # 'kernel_type' can only be 'linear' currently
    def __init__(self, C=5.0, tolerance=0.05, max_passes=10, kernel='linear'):
        self.C = C
        self.tolerance = tolerance
        self.max_passes = max_passes
        self.kernel_type = kernel
        self.gama = 0.05

    def fit(self, x, y):
        self.verify_data(x, y)

        self.x = x
        self.y = y
        self.n = len(y)

        self.alphas = np.zeros(self.n)
        self.b = 0
        passes_count = 0

        while passes_count < self.max_passes:
            num_changed_alphas = 0
            # for each instance
            for i in range(self.n):
                object_i = self.obj_func_x(self.x[i]) - self.y[i]
                if (self.y[i] * object_i < -self.tolerance and self.alphas[i] < self.C) or (
                            self.y[i] * object_i > self.tolerance and self.alphas[i] > 0):
                    rand_int = np.random.randint(low=0, high=10, size=2)
                    j = rand_int[0] if rand_int[0] != i else rand_int[1]

                    object_j = self.obj_func_x(self.x[j]) - self.y[j]
                    a_i, a_j = self.alphas[i], self.alphas[j]

                    # L and H
                    if self.y[i] != self.y[j]:
                        L = max(0.0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0.0, self.alphas[j] + self.alphas[i] - self.C)
                        H = min(self.C, self.alphas[j] + self.alphas[i])

                    if L == H:
                        continue

                    # small eta
                    h = 2 * self.kernel(self.x[i], self.x[j]) - \
                        self.kernel(self.x[i], self.x[i]) - \
                        self.kernel(self.x[j], self.x[j])
                    if h >= 0:
                        continue

                    # new alpha[j]
                    self.alphas[j] = self.alphas[j] - (self.y[j] * (object_i - object_j)) / h
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    elif L <= self.alphas[j] <= H:
                        self.alphas[j] = self.alphas[j]
                    elif self.alphas[j] < L:
                        self.alphas[j] = L

                    if abs(a_j - self.alphas[j]) < 10 ** -5:
                        continue

                    # new alpha[i]
                    self.alphas[i] = self.alphas[i] + self.y[i] * self.y[j] * (a_j - self.alphas[j])

                    # new b
                    b1 = self.b - object_i - self.y[i] * (self.alphas[i] - a_i) * self.kernel(self.x[i], self.x[i]) - \
                         self.y[j] * (self.alphas[j] - a_j) * self.kernel(self.x[i], self.x[j])
                    b2 = self.b - object_j - self.y[i] * (self.alphas[i] - a_i) * self.kernel(self.x[i], self.x[j]) - \
                         self.y[j] * (self.alphas[j] - a_j) * self.kernel(self.x[j], self.x[j])

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed_alphas += 1
            print("\tnum_changed_alphas = %d" % num_changed_alphas)
            if num_changed_alphas == 0:
                passes_count += 1
            else:
                passes_count = 0

        # save support vectors ids
        self.support_vectors = np.where(self.alphas > 0)[0]

    def predict(self, x):
        fx = [self.obj_func_x(xi) for xi in x]
        predictions = [-1 if fxi < 0 else 1 for fxi in fx]
        return predictions

    def obj_func_x(self, x_val):
        kernels_v = np.array([self.kernel(x_val, self.x[x]) if self.alphas[x] else 0 for x in range(self.n)])
        f_x = sum(self.alphas * self.y * kernels_v) + self.b
        return f_x

    def kernel(self, a, b):
        if self.kernel_type == 'linear':
            return sum(a * b)
        if self.kernel_type == 'rbf':
            return np.e**(-self.gama * np.linalg.norm(a - b,ord=2))
        else:
            raise Exception("Unsupported kernel")
            return None

    def verify_data(self, x, y):
        if len(x) != len(y):
            print("X and Y contain different number of samples!")
            exit(-1)
        if len(np.unique(y)) > 2:
            print("Currently only support 2 classes!")
            exit(-1)
        if 0 in y:
            y[y == 0] = -1
        return True

def main(cluster_std=2.5):
    mnist = fetch_mldata('MNIST original', data_home='./')
    mnist.keys()
    X_train, X_test, y_train, y_test = train_test_split(mnist.data / 255.0, mnist.target, test_size=0.002, random_state=42)
    data = np.array(X_train)
    labels = np.array(y_train)
    data_test = np.array(X_test)
    labels_test = np.array(y_test)
    print("Get mnist data successfully. ")

    num_kind_labels = 10
    svmArr = []
    for i in range(num_kind_labels):
        my_svm = SimpleSvm()
        svmArr.append(my_svm)
    labelsArr = []
    data_ = np.array(data[0:1000])
    labels_ = np.array(labels[0:1000])
    print(set(list(labels_)))
    for i in range(num_kind_labels):
        labels_i = np.array([1 if labels_[j] == i else 0 for j in range(len(labels_))])
        labelsArr.append(labels_i)
    for i in range(num_kind_labels):
        print("Start to fit svm model", i, ". ")
        svmArr[i].fit(data_, labelsArr[i])
        print("Fitting svm model", i, "successfully. ")
    predictArr = []
    for i in range(num_kind_labels):
        predictArr.append(svmArr[i].predict(data_test))
    predict = []
    for i in range(len(predictArr[0])):
        if predictArr[0][i] == 1:
            predict.append(0)
        elif predictArr[1][i] == 1:
            predict.append(1)
        elif predictArr[2][i] == 1:
            predict.append(2)
        elif predictArr[3][i] == 1:
            predict.append(3)
        elif predictArr[4][i] == 1:
            predict.append(4)
        elif predictArr[5][i] == 1:
            predict.append(5)
        elif predictArr[6][i] == 1:
            predict.append(6)
        elif predictArr[7][i] == 1:
            predict.append(7)
        elif predictArr[8][i] == 1:
            predict.append(8)
        else:
            predict.append(9)
    accuracy = metrics.accuracy_score(predict, labels_test)
    print(accuracy)

    with open("svm_model_1000linear", 'wb') as outf:
        outf.write(pickle.dumps(svmArr))

    # __svmArr__ = None
    # with open("svm_model_20", 'rb') as inf:
    #     __svmArr__ = pickle.loads(inf.read())
    # predictArr = []
    # for i in range(num_kind_labels):
    #     predictArr.append(__svmArr__[i].predict(data_test))
    # predict = []
    # for i in range(len(predictArr[0])):
    #     if predictArr[0][i] == 1:
    #         predict.append(0)
    #     elif predictArr[1][i] == 1:
    #         predict.append(1)
    #     elif predictArr[2][i] == 1:
    #         predict.append(2)
    #     elif predictArr[3][i] == 1:
    #         predict.append(3)
    #     elif predictArr[4][i] == 1:
    #         predict.append(4)
    #     elif predictArr[5][i] == 1:
    #         predict.append(5)
    #     elif predictArr[6][i] == 1:
    #         predict.append(6)
    #     elif predictArr[7][i] == 1:
    #         predict.append(7)
    #     elif predictArr[8][i] == 1:
    #         predict.append(8)
    #     else:
    #         predict.append(9)
    # accuracy = metrics.accuracy_score(predict, labels_test)
    # print(accuracy)

    # itr_num = 1
    # size = len(data) / itr_num
    # for iter in range(itr_num):
    #     print("iter", iter, ": ", iter * size, "to", (iter + 1) * size)
    #     labelsArr = []
    #     data_ = data[int(iter * size) : int((iter + 1) * size)]
    #     labels_ = labels[int(iter * size) : int((iter + 1) * size)]
    #     for i in range(num_kind_labels):
    #         labelsArr.append(np.array([1 if labels_[i] == i else 0 for i in range(len(labels_))]))
    #     for i in range(num_kind_labels):
    #         print("\tStart to fit svm model", i, ". ")
    #         svmArr[i].fit(data_, labelsArr[i])
    #         print("\tFitting svm model", i, "successfully. ")
    #     predictArr = []
    #     for i in range(num_kind_labels):
    #         predictArr.append(svmArr[i].predict(data_test))
    #     predict = []
    #     for i in range(len(predictArr[0])):
    #         if predictArr[0][i] == 1:
    #             predict.append(0)
    #         elif predictArr[1][i] == 1:
    #             predict.append(1)
    #         elif predictArr[2][i] == 1:
    #             predict.append(2)
    #         elif predictArr[3][i] == 1:
    #             predict.append(3)
    #         elif predictArr[4][i] == 1:
    #             predict.append(4)
    #         elif predictArr[5][i] == 1:
    #             predict.append(5)
    #         elif predictArr[6][i] == 1:
    #             predict.append(6)
    #         elif predictArr[7][i] == 1:
    #             predict.append(7)
    #         elif predictArr[8][i] == 1:
    #             predict.append(8)
    #         else:
    #             predict.append(9)
    #     accuracy = metrics.accuracy_score(predict, labels_test)
    #     print(accuracy)

if __name__ == '__main__':

    main()