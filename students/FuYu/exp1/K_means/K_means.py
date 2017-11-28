
# coding: utf-8

# In[1]:


from numpy import *

def loadData(file):
    dataset = []
    f = open(file)
    for line in f.readlines():
        cur = line.strip().split(",")
        flt = list(map(float, cur))
        dataset.append(flt)
    return dataset

def getDistance(va, vb):
#     两个点之间各个维度的数值差的平方之和开根，即欧氏距离
    return sqrt(sum(power(va-vb, 2)))




def getInit(datamat, k):
#     获取列数
    col = shape(datamat)[1]  
#     生成k行col列的零矩阵，k为设定的类数
    center = mat(zeros((k, col)))
    for i in range(col):
#         求得每列最大最小值，相减得到横纵坐标的scale，生成在scale范围内的随机数作为随机点的坐标
        minI = min(datamat[:,i])
        maxI = max(datamat[:,i])
        rangeI = float(array(maxI-minI))
        center[:,i] = minI +rangeI*random.rand(k, 1)
#         rangeI = float(max(array(datamat)[:,i])-minI)1
    return center


def K_means(datamat, k):
#     获得所有点数
    row = shape(datamat)[0]
#     new一个m行2列的空矩阵，第一列为对应的质心，第二列为对应的距离的平方
    assment = mat(zeros((row, 2)))
    center = getInit(datamat, k)
    isChange = True
    while isChange:   #如果还在变化
        isChange = False
#         遍历所有的点，对每个点都计算它和质心集中每个质心的距离
        for i in range(row):
#         最小的距离和对应的质心
            minDis = inf
            minIndex = -1
#         对每个点遍历全部k个质心
            for j in range(k):
#         center[j,:]得到的是center中第j行的行向量，[:,j]得到的是第j列的列向量
#         用两个数组的对应行向量求距离
                dis = getDistance(center[j,:], datamat[i,:])
                if dis<minDis:
                    minDis = dis
                    minIndex = j
#       assment[i,0]得到坐标i,0的点
            if assment[i,0] != minIndex:
                isChange = True;#点又发生了变化
            assment[i,:]=minIndex,minDis**2
#             每个点都通过寻找它们的质心来分成了k类，然后对属于同一类的点重新计算质心
        for cnt in range(k):
#         nonzero找到相等的点坐标,assment[;,0]找到第0列（存储index的列），和cnt比较，找到所有属于这个质心的点，nonzero[0]
            points = datamat[nonzero(assment[:,0].A==cnt)[0]]
#     axis = 0,压缩行，把求出的各列的均值存储到一行中
            center[cnt,:] = mean(points, axis = 0)
    return center, assment


def show(datamat, k, center, assment):
    from matplotlib import pyplot as plt
    num, dim =datamat.shape
    markp = ['or', 'ob', 'og', 'om', 'oy', '+r', 'oc', '<r', 'sr', 'pr']  
    for i in range(num):
        marki = int(assment[i,0])#mark是每一个点的质心序号（0，1，……，k-1）
        plt.plot(datamat[i,0], datamat[i,1], markp[marki])
    markc = ['Dk', 'Dk', 'Dk', 'Dk', 'Dk', '+b', 'Dk', 'Dk', '<b', 'pb']
    for i in range(k):
        plt.plot(center[i,0], center[i,1], markc[i], markersize = 8)
    plt.show()

def main():
    k = 2
    dataset = loadData("flame_co.txt")
    datamat = mat(dataset)
    theCenter,theAssment = K_means(datamat, k)
    print("初始质心坐标：")
    print(theCenter)
    print("分类数量：",k)
    show(datamat, k, theCenter, theAssment)
    
    k = 5
    dataset = loadData("agg_co.txt")
    datamat = mat(dataset)
    theCenter,theAssment = K_means(datamat, k)
    print("初始质心坐标：")
    print(theCenter)
    print("分类数量：",k)
    show(datamat, k, theCenter, theAssment)
        
if __name__ == '__main__':
    main()


