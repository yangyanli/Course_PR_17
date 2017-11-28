# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:50:23 2017

@author: Nancy
"""
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras import backend

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