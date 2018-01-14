from numpy import *
import time
import matplotlib.pyplot as plt


# calulate kernel value


def calcKernelValue(matrix_x, sample_x, kernelOption):#计算核函数
    kernelType = kernelOption[0]
    numSamples = matrix_x.shape[0]
    kernelValue = mat(zeros((numSamples, 1)))

    sigma = kernelOption[1]
    if sigma == 0:
         sigma = 1.0
    for i in xrange(numSamples):
         diff = matrix_x[i, :] - sample_x
         kernelValue[i] = exp(diff * diff.T *0.025)
    return kernelValue


# 得到核函数的矩阵
def calcKernelMatrix(train_x, kernelOption):
    numSamples = train_x.shape[0]
    kernelMatrix = mat(zeros((numSamples, numSamples)))
    for i in xrange(numSamples):
        kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
    return kernelMatrix


# svm的类
class SVMStruct:
    def __init__(self, dataSet, labels, C, toler, kernelOption):
        self.train_x = dataSet  # 每一行代表一个样本
        self.train_y = labels  # 相应的标签
        self.C = C  # 松弛变量
        self.toler = toler  # 终止条件迭代
        self.numSamples = dataSet.shape[0]  # 样本数量
        self.alphas = mat(zeros((self.numSamples, 1)))  # 所有样本的拉格朗日因子
        self.b = 0
        self.errorCache = mat(zeros((self.numSamples, 2)))
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)


       
# 计算αk的误差
def calcError(svm, alpha_k):
    output_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


# 在优化alpha k之后，更新alpha k的错误缓存
def updateError(svm, alpha_k):
    error = calcError(svm, alpha_k)
    svm.errorCache[alpha_k] = [1, error]


# 选择具有最大步骤的alpha j
def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i]  #标记为有效（已优化）
    print svm.errorCache
    candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[0]  
    print candidateAlphaList
    maxStep = 0;
    alpha_j = 0;
    error_j = 0

    # 用最大迭代步骤找到alpha，EJ-EI最大的j
    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_k - error_i) > maxStep:
                maxStep = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
                # 如果第一次进入这个循环，我们随机选择alpha j
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(random.uniform(0, svm.numSamples))
        error_j = calcError(svm, alpha_j)

    return alpha_j, error_j


# 用于优化alpha i和alpha j的内部循环
def innerLoop(svm, alpha_i):
    error_i = calcError(svm, alpha_i)

    ### 检查并提取违反KKT条件的alpha
    ## 满足KKT条件
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
    ## 违反KKT条件
    # 因为 y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, 所以
    # 1) 如果 y[i]*E_i < 0, so yi*f(i) < 1, 如果alpha <C，违反（alpha = C将是正确的）
    # 2) 如果 y[i]*E_i > 0, so yi*f(i) > 1, 如果alpha> 0，违反（alpha = 0将是正确的）
    # 3) 如果 y[i]*E_i = 0, so yi*f(i) = 1, 它是在边界上，不需要优化
    if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or \
                    (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

        # 第1步：选择alpha j
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # 步骤2：计算alpha j的边界L和H.
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0

            # 步骤3：计算eta（样本i和j的相似度）
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] \
              - svm.kernelMat[alpha_j, alpha_j]
        if eta >= 0:
            return 0

            # 第4步：更新alpha j
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta

        # 第5步：剪辑alpha j
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

            # 步骤6：如果alpha j不够动，就返回
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            updateError(svm, alpha_j)
            return 0

            # 第7步：优化aipha j后更新alpha i
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
                               * (alpha_j_old - svm.alphas[alpha_j])

        # 步骤8：更新阈值b
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                               * svm.kernelMat[alpha_i, alpha_i] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
               * svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                               * svm.kernelMat[alpha_i, alpha_j] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
               * svm.kernelMat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

            # 步骤9：在优化αi，j和b之后更新针对αi，j的错误缓存
        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        return 1
    else:
        return 0


        # 主要的训练程序

immm=20
def trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('rbf', 1.0)):
    # 计算训练时间
    startTime = time.time()

    # init数据结构为svm
    svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)

    print svm.kernelOpt[0]
    # 开始训练
    entireSet = True
    alphaPairsChanged = 0
    iterCount = 0
    # 迭代终止条件：
    # 条件1：达到最大迭代
    # 条件2：经过所有样本后没有发生alpha变化，
    # 换句话说，所有alpha（样本）都符合KKT条件
    while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0

        #更新所有训练示例中的alpha
        if entireSet:
            for i in xrange(svm.numSamples):
                alphaPairsChanged += innerLoop(svm, i)
            print '---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)
            iterCount += 1
            # 更新alpha的例子，其中alpha不是0而不是C（不在边界上）
        else:
            nonBoundAlphasList = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            for i in nonBoundAlphasList:
                alphaPairsChanged += innerLoop(svm, i)
            print '---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)
            iterCount += 1

            # 在所有示例和非边界示例上交替循环
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True

    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    global immm
    immm=immm+1
    im = bytes(immm)
  #  savetxt("/home/bpei/mssb/exp3/resultt"+im+".txt", svm.alphas )
   # savetxt("/home/bpei/mssb/exp3/x" + im + ".txt", svm.train_x)
  #  savetxt("/home/bpei/mssb/exp3/y" + im + ".txt", svm.train_y)
  #  savetxt("/home/bpei/mssb/exp3/b" + im + ".txt", svm.b)
    return svm


# 测试你的训练有素的svm模型给定的测试集
def testSVM(svm,test_x, test_y,irrr):
    test_x = mat(test_x)
    #print test_x
    test_y = mat(test_y)

    numTestSamples = test_x.shape[0]
    ir=bytes(irrr)
    #a =loadtxt("/home/bpei/mssb/exp3/resultt"+ir+".txt")
    #print a
    #b =loadtxt("/home/bpei/mssb/exp3/b" + ir + ".txt")
    #x =loadtxt("/home/bpei/mssb/exp3/x" + ir + ".txt")
    #y =loadtxt("/home/bpei/mssb/exp3/y" + ir + ".txt")

    #supportVectorsIndex = nonzero(mat(a).A > 0)[0]


    a = svm.alphas
    n=a.size
    b = svm.b
    x = svm.train_x
    y = svm.train_y
    #print supportVectorsIndex
    #supportVectorsIndex = argmax(supportVectorsIndex,0)
    #print supportVectorsIndex

    #print supportVectors
    #print supportVectorLabels
    #print supportVectorAlphas
    matchCount = 0
    predict=0
    #print x
    for i in xrange(numTestSamples):
        for j in xrange(n):
            if(a[j]> 0 and a[j] < 11):
                diff = x[j] - test_x[i, :]
                kernelValue = exp(diff * diff.T *0.025)
                predict += kernelValue.T * multiply(y[j], a[j])

        predict+=b

        #print sign(predict)
        if sign(predict) == sign(test_y[i]):
            matchCount += 1

    accuracy = float(matchCount) / numTestSamples
    #print accuracy
    return accuracy,sign(predict)
