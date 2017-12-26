# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:17:44 2017

@author: Nancy
"""  
import numpy as np 
import struct 
from sklearn import svm
from sklearn.decomposition import PCA 
from numpy import random
import time
from tqdm import tqdm
class DataUtils(object):
    """MNIST数据集加载
    输出格式为：numpy.array()    
    """
    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath
        
        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'    
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte
    
    def getImage(self):
        """
        将MNIST的二进制文件转换成像素特征数据
        """
        binfile = open(self._filename, 'rb') #以二进制方式打开文件
        buf = binfile.read() 
        binfile.close()
        index = 0
        numMagic,numImgs,numRows,numCols=struct.unpack_from(self._fourBytes2,\
                                                                    buf,\
                                                                    index)
        index += struct.calcsize(self._fourBytes)
        images = []
        for i in range(numImgs):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            for j in range(len(imgVal)):
                if imgVal[j] > 1:
                    imgVal[j] = 1
            images.append(imgVal)
        return np.array(images)
        
    def getLabel(self):
        """
        将MNIST中label二进制文件转换成对应的label数字特征
        """
        binFile = open(self._filename,'rb')
        buf = binFile.read()
        binFile.close()
        index = 0
        magic, numItems= struct.unpack_from(self._twoBytes2, buf,index)
        index += struct.calcsize(self._twoBytes2)
        labels = [];
        for x in range(numItems):
            im = struct.unpack_from(self._labelByte2,buf,index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)


class SVM:  
    def Gauss_kernel(x,z,gamma = 0.025):  
        return np.exp(-np.sum((x-z)**2)*gamma)  
    def __init__(self,train_X,train_y,C=11,tol=0.001,kernel=Gauss_kernel):  
        self.train_X=np.array(train_X)  
        self.train_y=np.array(train_y).flatten("F")   
        self.tol=tol  
        self.M,self.N=self.train_X.shape  
        self.C=C  
        self.kernel=kernel  
        self.alpha=np.zeros((1,self.M)).flatten("F")  
        self.supportVec=[]  
        self.b=0  
        self.E=np.zeros((1,self.M)).flatten("F")  
    def fitKKT(self,i):  
        if ((self.train_y[i]*self.E[i]<-self.tol) and (self.alpha[i]<self.C)) or (((self.train_y[i]*self.E[i]>self.tol)) and (self.alpha[i]>0)):  
            return False  
        return True   
  
    def select(self,i):  
        pp=np.nonzero((self.alpha>0))[0]  #返回alpha中非零元素的位置
        if (pp.size>0):  
            j=self.findMax(i,pp)  
        else:  
            j=self.findMax(i,range(self.M))  
        return j  
  
    def randJ(self,i):  
        j=random.sample(range(self.M))  
        return j[0]  
    def findMax(self,i,ls):  
        ansj=-1  
        maxx=-1  
        self.updateE(i)  
        for j in ls:  
            if i==j:continue  
            self.updateE(j)  
            tempE=np.abs(self.E[i]-self.E[j])  
            if tempE>maxx:  
                maxx=tempE  
                ansj=j  
        if ansj==-1:  
            return self.randJ(i)  
        return ansj  
  
    def InerLoop(self,i,threshold):  
        j=self.select(i)  
        self.updateE(j)  
        self.updateE(i)  
        if (self.train_y[i]==self.train_y[j]):  
            L=max(0,self.alpha[i]+self.alpha[j]-self.C)  
            H=min(self.C,self.alpha[i]+self.alpha[j])  
        else:  
            L=max(0,self.alpha[j]-self.alpha[i])  
            H=min(self.C,self.C+self.alpha[j]-self.alpha[i])  
  
        a2_old=self.alpha[j]  
        a1_old=self.alpha[i]  
          
        K12=self.kernel(self.train_X[i],self.train_X[j])  
        eta=2-2*self.kernel(self.train_X[i],self.train_X[j]) 
        if eta==0:  
            return True  
          
        self.alpha[j]=self.alpha[j]+self.train_y[j]*(self.E[i]-self.E[j])/eta  
          
        if self.alpha[j]>H:  
            self.alpha[j]=H  
        elif self.alpha[j]<L:  
            self.alpha[j]=L  
  
        if np.abs(self.alpha[j]-a2_old)<threshold:  
            return True  
        
        self.alpha[i]=self.alpha[i]+self.train_y[i]*self.train_y[j]*(a2_old-self.alpha[j])  
        b1_new=self.b-self.E[i]-self.train_y[i]*(self.alpha[i]-a1_old)-self.train_y[j]*K12*(self.alpha[j]-a2_old)  
        b2_new=self.b-self.E[j]-self.train_y[i]*K12*(self.alpha[i]-a1_old)-self.train_y[j]*(self.alpha[j]-a2_old)  

        if self.alpha[i]>0 and self.alpha[i]<self.C:
            self.b=b1_new  
        elif self.alpha[j]>0 and self.alpha[j]<self.C:
            self.b=b2_new  
        else:   
            self.b=(b1_new+b2_new)/2  
  
        self.updateE(j)  
        self.updateE(i)  
        return False  
  
    def updateE(self,i): 
        '''
        self.E[i]=0  
        kernels = self.kernel(self.train_X[i],self.train_X)
        self.E[i] = np.sum(self.alpha*self.train_y*kernels)
        self.E[i]+=self.b-self.train_y[i]
        '''
        self.E[i]=0  
        for t in range(self.M):  
        #for t in range(self.M):  
            self.E[i]+=self.alpha[t]*self.train_y[t]*self.kernel(self.train_X[i],self.train_X[t])  
        self.E[i]+=self.b-self.train_y[i] 
        
    def train(self,maxii=30 ,threshold=0.001):  
        flag=False  
        for i in tqdm(range(self.M)): 
            self.updateE(i)
        for i in tqdm(range(maxii)):  
            flag=True  
            temp_supportVec=np.nonzero((self.alpha>0))[0]
            for i in temp_supportVec:
                #print("1")
                self.updateE(i)  
                if (not self.fitKKT(i)):  
                    flag= flag and self.InerLoop(i,threshold)
            if (flag):
                for i in range(self.M):
                    self.updateE(i)  
                    if (not self.fitKKT(i)):  
                        flag= flag and self.InerLoop(i,threshold)  
            if(flag):
                break
            #print ("the %d-th iter is running" % ii)  
        self.supportVec=np.nonzero((self.alpha>0))[0]  
    def predict(self,test_X):
        test_X=np.array(test_X)  
        y=[]  
        for i in range(test_X.shape[0]): 
            w=0  
            for t in self.supportVec:  
                w+=self.alpha[t]*self.train_y[t]*self.kernel(self.train_X[t],test_X[i]).flatten("F")  
            w+=self.b
            y.append(w) 
        py=np.array(y).flatten("F")   
        return py
 

def GetData():
    trainfile_X = 'train-images-idx3-ubyte/train-images.idx3-ubyte'
    trainfile_y = 'train-labels-idx1-ubyte/train-labels.idx1-ubyte'
    testfile_X = 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
    testfile_y = 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()
    
    print(train_X)
    print(train_X.shape)
    print(train_y)
    print(test_X)
    print(test_y)
    return train_X, train_y, test_X, test_y 
 
 
def testSVM(traina_X,traina_y, testa_X, testa_y): 
    print("first")
    print(testa_y)
    train_a,train_b=traina_X.shape
    test_a,test_b=testa_X.shape
    temparr = []
    ans = []
    arr = []
    tempy = []
    ty=[]
    ty=traina_y
    tempy = np.zeros(np.array(traina_y).shape)
    error = 0
    for i in range (0,1):
        print(i)
        for j in range(0,train_a):
            if(traina_y[j]==i):
                tempy[j]=1
            else:
                tempy[j]=-1
        svms=SVM(traina_X,tempy)
        svms.train()
        temparr = svms.predict(testa_X)
        if(i == 0):
            ans = np.zeros(np.array(temparr).shape)
            arr = temparr
            ans[temparr>0]=1
            ans[temparr<0]=-1
        else:
            for j in range(0,test_a):
                if(temparr[j]>arr[j]):
                    ans[j]=i
                    arr[j]=temparr[j]
        print(ans)
        error=0
        for k in range(0,train_a):
            if(tempy[k]!=ans[k]):
              error += 1  
        print ("the error_case is  ",error)
    print(ans)
    print("final")
    print(testa_y)
    kerror=0
    for k in range(0,train_a):
        if(testa_y[k]!=ans[k]):
            kerror += 1  
    print ("the error_case is  ",kerror)
    print ("the error_case is  ",np.sum(ans!=np.array(ty)))  
    print ("SupportVectors = %d "%len(svms.supportVec))
    
def main():
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train_X, train_y, test_X, test_y = GetData()
    pca=PCA(n_components=5)  
    train_X=pca.fit_transform(train_X)
    test_X=pca.fit_transform(test_X)
    print(train_X.shape)
    print(test_X.shape)
    
    testSVM(trian_X,train_y, test_X, test_y)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
def KSVM():
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train_X, train_y, test_X, test_y = GetData()
    pca=PCA(n_components=5)  
    train_X=pca.fit_transform(train_X)
    test_X=pca.fit_transform(test_X)
    clf = svm.SVC(C=11,gamma=0.025)
    clf.fit(test_X, test_y)
    predictions = clf.predict(test_X)
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_y))
    print("The accuracy is %f "%(num_correct/len(test_y)))
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
if __name__ == "__main__":
    main()
    KSVM()
