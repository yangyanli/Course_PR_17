# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:32:20 2017

@author: Nancy
"""
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from skimage import io,transform,color
import matplotlib.pyplot as plt
from pylab import imsave ,subplot

img=io.imread('C:/Users/Nancy/Desktop/模式识别/ex2/test/pic6.png')
img_gray=color.rgb2gray(img)
rows,cols=img_gray.shape
UNCLASSIFIED = False
NOISE = 0

def LoadDataSet():
    dataset = []
    x = 0
    for i in range(rows):
        for j in range(cols):
            if(img_gray[i][j]==True):
                xx=[i,j]
                dataset.append(xx)
                x=x+1
    return dataset

def expand(point,ID,result,dataset,num):
    result[point] = ID
    x1 = dataset[point][0]
    x2 = dataset[point][0]
    y1 = dataset[point][1]
    y2 = dataset[point][1]
    seeds = []
    for i in range(point,num):
        if (dataset[i][0]==dataset[point][0] and result[i]==UNCLASSIFIED and dataset[i][1]==dataset[point][1]-1):
            seeds.append(i)
            result[i] = ID
        elif(dataset[i][0]==dataset[point][0] and dataset[i][1]==dataset[point][1]+1 and result[i]==UNCLASSIFIED):
            seeds.append(i)
            result[i] = ID
        elif(dataset[i][0]==dataset[point][0]-1 and dataset[i][1]==dataset[point][1] and result[i]==UNCLASSIFIED):
            seeds.append(i)
            result[i] = ID
        elif(dataset[i][0]==dataset[point][0]+1 and dataset[i][1]==dataset[point][1] and result[i]==UNCLASSIFIED):
            seeds.append(i)
            result[i] = ID
        elif(dataset[i][0]==dataset[point][0]+1 and dataset[i][1]==dataset[point][1]+1 and result[i]==UNCLASSIFIED):
            seeds.append(i)
            result[i] = ID
        elif(dataset[i][0]==dataset[point][0]+1 and dataset[i][1]==dataset[point][1]-1 and result[i]==UNCLASSIFIED):
            seeds.append(i)
            result[i] = ID
        elif(dataset[i][0]==dataset[point][0]-1 and dataset[i][1]==dataset[point][1]+1 and result[i]==UNCLASSIFIED):
            seeds.append(i)
            result[i] = ID
        elif(dataset[i][0]==dataset[point][0]-1 and dataset[i][1]==dataset[point][1]-1 and result[i]==UNCLASSIFIED):
            seeds.append(i)
            result[i] = ID
        elif(dataset[i][0]>=dataset[point][0]+2):
            break
    while len(seeds) > 0:
        currentpoint = seeds[0]
        if(dataset[currentpoint][0] < x1):
            x1 = dataset[currentpoint][0]
        elif(dataset[currentpoint][0] > x2):
            x2 = dataset[currentpoint][0]
        if(dataset[currentpoint][1] < y1):
            y1 = dataset[currentpoint][1]
        elif(dataset[currentpoint][1] > y2):
            y2 = dataset[currentpoint][1]
        for i in range(num):
            if (dataset[i][0]==dataset[currentpoint][0] and dataset[i][1]==dataset[currentpoint][1]-1 and result[i]==UNCLASSIFIED):
                seeds.append(i)
                result[i] = ID
            elif(dataset[i][0]==dataset[currentpoint][0] and dataset[i][1]==dataset[currentpoint][1]+1 and result[i]==UNCLASSIFIED):
                seeds.append(i)
                result[i] = ID
            elif(dataset[i][0]==dataset[currentpoint][0]-1 and dataset[i][1]==dataset[currentpoint][1] and result[i]==UNCLASSIFIED):
                seeds.append(i)
                result[i] = ID
            elif(dataset[i][0]==dataset[currentpoint][0]+1 and dataset[i][1]==dataset[currentpoint][1] and result[i]==UNCLASSIFIED):
                seeds.append(i)
                result[i] = ID
            elif(dataset[i][0]==dataset[point][0]+1 and dataset[i][1]==dataset[point][1]+1 and result[i]==UNCLASSIFIED):
                seeds.append(i)
                result[i] = ID
            elif(dataset[i][0]==dataset[point][0]+1 and dataset[i][1]==dataset[point][1]-1 and result[i]==UNCLASSIFIED):
                seeds.append(i)
                result[i] = ID
            elif(dataset[i][0]==dataset[point][0]-1 and dataset[i][1]==dataset[point][1]+1 and result[i]==UNCLASSIFIED):
                seeds.append(i)
                result[i] = ID
            elif(dataset[i][0]==dataset[point][0]-1 and dataset[i][1]==dataset[point][1]-1 and result[i]==UNCLASSIFIED):
                seeds.append(i)
                result[i] = ID
            elif(dataset[i][0]>=dataset[currentpoint][0]+2):
                break
        seeds = seeds[1:]
    return result, ID+1,x1,x2,y1,y2


     
def classify(dataset,num):
    ID = 1
    result = [UNCLASSIFIED]*num
    x = []
    y = []
    box = []
    for i in range(num):
        if(result[i]==UNCLASSIFIED):
            result,ID,a1,a2,a3,a4 = expand(i,ID,result,dataset,num)
            #print(a1,a2,a3,a4)
            a = [a1,a1,a2,a2,a1]
            b = [a3,a4,a4,a3,a3]
            c = [a1,a2,a3,a4]
            x.append(a)
            y.append(b)
            box.append(c)
    return result,ID-1,x,y,box

def CreateModel():     
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    model = Sequential()
    
    classes = 10
    batch_size = 32
    epochs = 8
    
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')#keras输入格式中通道数在后面并把数据变成float格式
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    input_shape = (28, 28, 1)#28*28像素的灰度图
    
    x_train /= 255
    x_test /= 255
    
    y_train = keras.utils.np_utils.to_categorical(y_train, classes)#类别变成二进制
    y_test = keras.utils.np_utils.to_categorical(y_test, classes)
    
    model.add(Conv2D(32,kernel_size = (3,3),activation='relu',input_shape=input_shape,data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128,kernel_size = (3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    
    model.compile(loss=keras.metrics.categorical_crossentropy,optimizer=keras.optimizers.Adagrad(),metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=2)
    print('Final loss:', score[0])
    print('Final accuracy:', score[1])
    return model
    
def pict(model):
    img1=io.imread('C:/Users/Nancy/Desktop/模式识别/ex2/now.jpg')
    img_gray=color.rgb2gray(img1)
    rows,cols=img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if (img_gray[i,j]<=0.5):
                img_gray[i,j]=False
            else:
                img_gray[i,j]=True
    #io.imshow(img_gray)
    img1 = img_gray.reshape(1,28,28,1)
    r = model.predict(img1, batch_size=32, verbose=0)
    print(r)
    result = model.predict_classes(img1, batch_size=32, verbose=0)
    print(result[0])
    return result[0]


def main():
    num = 0
    for i in range(rows):
        for j in range(cols):
            if (img_gray[i,j]<=0.5):
                img_gray[i,j]=True
            else:
                img_gray[i,j]=False
    io.imshow(img_gray)
    dataset = LoadDataSet()
    print(len(dataset))
    num = len(dataset)
    result, ID, x, y,box = classify(dataset,num)
    print(ID)
    model = CreateModel()
    j=1
    predict_class = []
    for i in range(ID):
        if(y[i][1]-y[i][0]>200 or x[i][2]-x[i][0]>200):
            continue
        if(y[i][1]-y[i][0]>10 or x[i][2]-x[i][0]>10):
            print(j)
            length = box[i][1] - box[i][0]
            xmiddle = (box[i][3] + box[i][2])
            xa = xmiddle - length
            xb = xmiddle + length
            xa = int(xa/2)-5
            xb = int(xb/2)+5
            ya = box[i][0]-5
            yb = box[i][1]+5
            #subplot(9,4,j)
            j=j+1
            if(xa < 0):
                xa = 0
            if(ya < 0):
                ya = 0
            pic = (img_gray[ya:yb,xa:xb])
            pic = transform.resize(pic, (28, 28))
            imsave('C:/Users/Nancy/Desktop/模式识别/ex2/'+str(j)+'.jpg',pic)
            imsave('C:/Users/Nancy/Desktop/模式识别/ex2/now.jpg',pic)
            re = pict(model)
            predict_class.append(re)
            #plt.imshow(pic)
            #plt.show()
            #io.imshow(pic)
            
    print(predict_class)
if __name__ == '__main__':
    main()
    plt.show()





















