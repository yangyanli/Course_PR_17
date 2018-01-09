import numpy as np
import gzip
import time

#按32位读取
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def get_images(filename, is_valbinary, is_matrix):

    f = gzip.open(filename, 'rb')
    magic = _read32(f)
    num_images = _read32(f)
    rows = _read32(f)
    cols = _read32(f)
    #print(magic, num_images, rows, cols)
    buf = f.read(rows*cols*num_images)
    data = np.frombuffer(buf, dtype=np.uint8)

    if(is_matrix):
        data = data.reshape(num_images, rows*cols)
    else:
        data = data.reshape(num_images, rows, cols)

    if(is_valbinary):
        return np.minimum(data, 1)
    else:
        return data

def get_labels(filename):
    f = gzip.open(filename, 'rb')
    magic = _read32(f)
    num_items = _read32(f)
    buf = f.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    return labels

def knn(newInput, dataSet, labels, k):

    numSamples = dataSet.shape[0]
    init_shape = newInput.shape[0]
    newInput = newInput.reshape(1, init_shape)

    # 将数组newInput重复(numSamples, 1)次，构成一个新的数组
    diff = np.tile(newInput, (numSamples, 1)) - dataSet
    squDiff = diff ** 2
    squaredDist = np.sum(squDiff, axis=1)
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)

    classCount = {} #定义字典用来存放元素
    for i in range(k):
        #选取距离最小的k个点
        votrLabel = labels[sortedDistIndices[i]]
        classCount[votrLabel] = classCount.get(votrLabel, 0) + 1

    #the max class
    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if(value > maxCount):
            maxCount = value
            maxIndex = key

    return maxIndex

#加载数据集，此路径是本地mnist数据集所在文件夹
def loadData():

    train_x = get_images('F:\Git\Course_PR_17\experiment1\data\MNIST/t10k-images-idx3-ubyte.gz', True, True)
    train_y = get_labels('F:\Git\Course_PR_17\experiment1\data\MNIST/t10k-labels-idx1-ubyte.gz')
    test_x = get_images('F:\Git\Course_PR_17\experiment1\data\MNIST/train-images-idx3-ubyte.gz', True, True)
    test_y = get_labels('F:\Git\Course_PR_17\experiment1\data\MNIST/train-labels-idx1-ubyte.gz')

    return train_x, train_y, test_x, test_y

#the main function
def testHWClass():

    print('Loading dataset')
    train_x, train_y, test_x, test_y = loadData()
    #train_x = extract_images('F:\Git\Course_PR_17\experiment1\data\MNIST/t10k-images-idx3-ubyte.gz', True, True)
    #train_y = extract_labels('F:\Git\Course_PR_17\experiment1\data\MNIST/t10k-labels-idx1-ubyte.gz')
    #test_x = extract_images('F:\Git\Course_PR_17\experiment1\data\MNIST/train-images-idx3-ubyte.gz', True, True)
    #test_y = extract_labels('F:\Git\Course_PR_17\experiment1\data\MNIST/train-labels-idx1-ubyte.gz')

    print('Training...')
    print('Testing...')
    t1 = time.time()
    numTestSamples = test_x.shape[0]
    errorCount = 0
    test_num = int(numTestSamples / 10)
    #test_num = 1000

    for i in range(1, test_num + 1):
        predict = knn(test_x[i], train_x, train_y, 5)
        if(predict != test_y[i]):
            errorCount += 1
        if(i % 100 == 0):
            print('finish ' + str(i) + ' images')

    error = float(errorCount) / test_num
    accuracy = 1 - error

    t2 = time.time()
    runningtime = t2 - t1;
    print('running time: %.2fmin,%.4fs.' % (runningtime // 60, runningtime % 60))

    print('The result:')
    print('  errorCount: %d' % errorCount)
    print('  accuracy: %.4f' % accuracy)

if __name__ == '__main__':
    testHWClass()