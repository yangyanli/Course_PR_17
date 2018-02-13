from numpy import *
from time import sleep

def loadDataSet(fileName):#testSet.txt
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat, labelMat

def selectJrand(i, m):
	j = i
	while(j == i):
		j = int(random.uniform(0, m))
	return j

def clipAlpha(aj, H, L):
	aj = min(aj, H)
	aj = max(aj, L)
	return aj

def smoSimple(dataMatIn, classLabels, C, eps, maxIter):
	dataMat = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	b = 0
	m, n = shape(dataMat)
	alphas = mat(zeros((m, 1)))
	iter = 0
	while iter < maxIter:
		alphaPairsChanged = 0
		for i in range(m):
			fXi = float(multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
			Ei = fXi - float(labelMat[i])
			if ((labelMat[i] * Ei < -eps) and (alphas[i] < C) or (labelMat[i] * Ei > eps) and alphas[i] > 0):
				j = selectJrand(i, m)
				fXj = float(multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
				Ej = fXj - float(labelMat[j])
				alphaIOld = alphas[i].copy()
				alphaJOld = alphas[j].copy()
				if (labelMat[i] != labelMat[j]):
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:
					L = max(0, alphas[j] - alphas[i] - C)
					H = min(C, alphas[j] + alphas[i])
				if(L == H):
					print "L = H"
					continue
				eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :] * dataMat[i, :].T - dataMat[j, :] * dataMat[j, :].T
				if (eta >= 0):
					print "eta >= 0"
					continue
				alphas[j] -= labelMat[j] * (Ei - Ej) / eta
				alphas[j] = clipAlpha(alphas[j], H, L)
				if(abs(alphas[j] - alphaJOld) < 0.00001):
					print "j moving too small"
					continue
				alphas[i] += labelMat[j] * labelMat[i] * (alphaJOld - alphas[j])
				b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIOld) * dataMat[i, :] * dataMat[i, :].T -\
				labelMat[j] * (alphas[j] - alphaJOld) * dataMat[i, :] * dataMat[j, :].T
				b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIOld) * dataMat[i, :] * dataMat[j, :].T - \
				labelMat[j] * (alphas[j] - alphaJOld) * dataMat[j, :] * dataMat[j, :].T
				if (alphas[i] > 0) and (alphas[i] < C):
					b = b1
				elif (alphas[j] > 0) and (alphas[j] < C):
					b = b2
				else:
					b = (b1 + b2) / 2.0
				alphaPairsChanged += 1 
				print "iter: %d i: %d, pairs changed %d" %(iter, i, alphaPairsChanged)
		if alphaPairsChanged == 0:
			iter += 1
		else:
			iter = 0
		print "total iter: %d" % maxIter
	return b, alphas

class optType:#the data structure that stores major values of the params
	def __init__(self, dataMatIn, classLabels, C, eps, kTup):
		self.X = dataMatIn
		self.Y = classLabels
		self.C = C#constraint: 0 <= alpha <= C
		self.eps = eps#tolerance
		self.m = shape(dataMatIn)[0]#number of data points
		self.alphas = mat(zeros((self.m, 1)))
		self.b = 0
		self.E = mat(zeros((self.m, 2)))#errors; E[i][0]-is valid; E[i][1]-error value
		self.K = mat(zeros((self.m, self.m)))#kernel
		for i in range(self.m):
			self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def calcEi(data, i):
	#fXi = float(multiply(data.alphas, data.Y).T * (data.X * data.X[i, :].T) + data.b)
	fXi = float(multiply(data.alphas, data.Y).T * data.K[:, i] + data.b)
	#fXi = wx+b; w = sum(alpya*y*x)
	return fXi - float(data.Y[i])

def selectJ(data, i, Ei):#choose the second alpha
	maxDeltaE = -1000; idx = 0
	Ej = 0
	data.E[i] = [1, Ei]
	validEs = nonzero(data.E[:, 0].A)[0]#.A=> as an ndarray; indices of valid Es
	if len(validEs) > 1:
		for k in validEs:#find the one the maximizes delta E
			if k == i: continue
			Ek = calcEi(data, k)
			deltaE = abs(Ei - Ek)
			if deltaE > maxDeltaE:
				idx = k
				maxDeltaE = deltaE
				Ej = Ek
		return idx, Ej
	else:#validEs <= 1
		j = selectJrand(i, data.m)
		Ej = calcEi(data, j)
		return j, Ej

def updateEi(data, i):
	Ei = calcEi(data, i)
	data.E[i] = [1, Ei]

def innearL(data, i):
	Ei = calcEi(data, i)
	if ((data.Y[i] * Ei < -data.eps) and (data.alphas[i] < data.C)) or ((data.Y[i] * Ei > data.eps) and (data.alphas[i] > 0)):
		j, Ej = selectJ(data, i, Ei)
		alphaIOld = data.alphas[i].copy()
		alphaJOld = data.alphas[j].copy()
		if(data.Y[i] != data.Y[j]):
			L = max(0, data.alphas[j] - data.alphas[i])
			H = min(data.C, data.C + data.alphas[j] - data.alphas[i])
		else:
			L = max(0, data.alphas[j] + data.alphas[i] - data.C)
			H = min(data.C, data.alphas[j] + data.alphas[i])
		if L == H:
			#print "L == H"
			return 0
		#eta = 2.0 * data.X[i, :] * data.X[j, :].T - data.X[i, :] * data.X[i, :].T - data.X[j, :] * data.X[j, :].T
		eta = 2.0 * data.K[i, j] - data.K[i, i] - data.K[j, j]
		if eta >= 0:
			#print "eta >= 0"
			return 0
		data.alphas[j] -= data.Y[j] * (Ei - Ej) / eta
		data.alphas[j] = clipAlpha(data.alphas[j], H, L)
		updateEi(data, j)
		if abs(data.alphas[j] - alphaJOld) < 0.00001:
			#print "j not moving enough"
			return 0
		data.alphas[i] += data.Y[j] * data.Y[i] * (alphaJOld - data.alphas[j])
		updateEi(data, i)
		#b1 = data.b - Ei - data.Y[i] * (data.alphas[i] - alphaIOld) * data.X[i, :] * data.X[i, :].T - \
		#	data.Y[j] * (data.alphas[j] - alphaJOld) * data.X[i, :] * data.X[j, :].T
		b1 = data.b - Ei - data.Y[i] * (data.alphas[i] - alphaIOld) * data.K[i, i] - \
			data.Y[j] * (data.alphas[j] - alphaJOld) * data.K[i, j]
		#b2 = data.b - Ej - data.Y[i] * (data.alphas[i] - alphaIOld) * data.X[i, :] * data.X[j, :].T - \
		#	data.Y[j] * (data.alphas[j] - alphaJOld) * data.X[j, :] * data.X[j, :].T
		b2 = data.b - Ej - data.Y[i] * (data.alphas[i] - alphaIOld) * data.K[i, j] - \
			data.Y[j] * (data.alphas[j] - alphaJOld) * data.K[j, j]
		if (data.alphas[i] > 0) and (data.alphas[i] < data.C): data.b = b1
		elif (data.alphas[j] > 0) and (data.alphas[j] < data.C): data.b = b2

		else: data.b = (b1 + b2) * 0.5
		return 1
	else:
		return 0

def smoPlatt(dataMatIn, classLabels, C, eps, maxIter, kTup=('lin', 0)):
	data = optType(mat(dataMatIn), mat(classLabels).transpose(), C, eps, kTup)#classLabels=>column vector
	iter = 0
	wholeSet = 1#boolean
	alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (wholeSet == 1)):
		alphaPairsChanged = 0
		if wholeSet == 1:
			for i in range(data.m):
				alphaPairsChanged += innearL(data, i)
				#print "full set, iter: %d i: %d, %d pairs changed" % (iter, i, alphaPairsChanged)
		else:
			nonBoundIs = nonzero((data.alphas.A > 0) * (data.alphas.A < C))[0]
			for i in nonBoundIs:
				alphaPairsChanged += innearL(data, i)
				#print "not bound, iter: %d, i: %d, %d pairs changed" % (iter, i, alphaPairsChanged)
		iter += 1
		if wholeSet == 1: wholeSet = 0
		elif alphaPairsChanged == 0: wholeSet = 1
		#print "iter number is %d" % iter
	return data.b, data.alphas

def calcWs(alphas, dataArr, classLabels):
	X = mat(dataArr)
	Y = mat(classLabels).transpose()
	m, n = shape(X)
	W = zeros((n, 1))
	for i in range(m):
		W += multiply(alphas[i] * Y[i], X[i, :].T)
	return W

def kernelTrans(X, A, kTup):
	m, n = shape(X)
	K = mat(zeros((m, 1)))
	if kTup[0] == 'lin': K = X * A.T#linear
	elif kTup[0] == 'rbf':#radial basis funciton
		for i in range(m):
			deltaRow = X[i, :] - A
			K[i] = deltaRow * deltaRow.T
		K = exp(K / (1 - kTup[1] * kTup[1]))#element-wise division
	else:
		print "error: the kernel is not supported"
	return K

def img2vec(fileName):
	fr = open(fileName)
	#print fileName
	res = zeros((1, 784))
	for i in range(28):
		line = fr.readline()
		for j in range(28):
			res[0, i * 28 + j] = int(line[j])
	return res

def loadImg(dirName, digit):
	from os import listdir
	trainingList = listdir(dirName)
	m = len(trainingList)
	X = zeros((m, 784))
	Y = []
	D = []
	for i in range(m):
		filename = trainingList[i].split('.')[0] #remove '.txt'
		number = int(filename.split('_')[0])
		if number == digit: Y.append(-1)
		else: Y.append(1)
		D.append(number)
		X[i, :] = img2vec('%s/%s' % (dirName, trainingList[i]))
	return X, Y, D


def work(kTup=('rbf', 20), trainingDir='trainingDigits', testDir='testDigits'):
	b = {}
	alphas = {}
	svIdx = {}
	svX = {}
	svY = {}
	for d in range(10):
		print "for number %d:" % (d)
		dataArr, labelArr, D = loadImg(trainingDir, d)
		print "\tdata loaded"
		b[d], alphas[d] = smoPlatt(dataArr, labelArr, 200, 0.0001, 1000, kTup)
		print "\ttraining finished"
		X = mat(dataArr)
		Y = mat(labelArr).transpose()
		svIdx[d] = nonzero(alphas[d].A > 0)[0]
		svX[d] = X[svIdx[d]]
		svY[d] = Y[svIdx[d]]
		print "\t%d support vectors in total" % shape(svX[d])[0]
		m, n = shape(X)
		errCnt = 0
		for i in range(m):
			kernelEval = kernelTrans(svX[d], X[i, :], kTup)
			predict = kernelEval.T * multiply(svY[d], alphas[d][svIdx[d]]) + b[d]
			if sign(predict) != sign(Y[i]): errCnt += 1
		print "\ttraining accuracy: %f" % (1.0 - float(errCnt)/m)

	dataArr, labelArr, DArr = loadImg(trainingDir, 0)
	print "test data loaded"
	X = mat(dataArr)
	Y = mat(labelArr).transpose()
	D = mat(DArr)
	m, n = shape(X)
	errCnt = 0
	for i in range(m):
		ck = 1
		for d in range(10):
			kernelEval = kernelTrans(svX[d], X[i, :], kTup)
			predict = kernelEval.T * multiply(svY[d], alphas[d][svIdx[d]]) + b[d]
			if ((predict[0, 0] < 0) and (d != D[0, i])) or ((predict[0, 0] > 0) and (d == D[0, i])): ck = 0
		if ck == 0: errCnt += 1
	print "\ttesting accuracy: %f" % (1.0 - float(errCnt)/m)