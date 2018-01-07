#coding=utf-8
#!/usr/bin/env python

import numpy as np
import struct
import math
import random
import tempfile
import string

def loadImage(filename = "mnist/train-images-idx3-ubyte"):
    print("Loading imgs...")
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    # I 表示unsigned int '>'表示大端存储
    header = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    imgNum = header[1]
    width = header[2]
    height = header[3]
    # print("imgNum = ",imgNum," width = ",width," height = ",height)
    bits = imgNum * width * height
    #B表示Unsigned char
    bitsString = '>' + str(bits) + 'B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    # print(type(imgs))
    imgs = np.reshape(imgs, [imgNum, width * height])
    print("Finish load imgs")
    # print(np.shape(imgs))
    return imgs

def loadLabel(filename = "mnist/train-labels-idx1-ubyte"):
    print("Loading labels...")
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    header = struct.unpack_from('>II', buffers, 0)
    imgNum = header[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    print("Finish load labels")
    return labels

if __name__ == "__main__":
    imgs = loadImage()
    labels = loadLabel()
    print(imgs[1])

    # imgs = loadImage("t10k-images.idx3-ubyte")
    # labels = loadLabel("t10k-labels.idx1-ubyte")