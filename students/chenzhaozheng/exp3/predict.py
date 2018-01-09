# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import mxnet as mx
from mxnet import nd,image,autograd,init
from mxnet.gluon import nn,Trainer
from CapsLayers import DigitCaps, PrimaryConv, Length
from mxnet.gluon.loss import L2Loss
import cv2
import numpy as np
def CapsNet(batch_size, ctx):

    net = nn.Sequential()
    with net.name_scope():

        net.add(nn.Conv2D(channels=256, kernel_size=9, strides=1, padding=(0,0), activation='relu'))
        net.add(PrimaryConv(dim_vector=8, n_channels=32, kernel_size=9, strides=2, context=ctx, padding=(0,0)))
        net.add(DigitCaps(num_capsule=10, dim_vector=16, context=ctx))
        net.add(Length())

    net.initialize(ctx=ctx, init=init.Xavier())
    return net

ctx = mx.cpu()
net = CapsNet(batch_size=1,ctx=ctx)
net.collect_params().load(filename='params_net',ctx=ctx)

img_path = []
img_path.append('./images/27.jpg')
img_path.append('./images/28.jpg')
img_path.append('./images/49.jpg')
img_path.append('./images/60.jpg')
img_path.append('./images/67.jpg')
img_path.append('./images/68.jpg')
img_path.append('./images/87.jpg')

img = []
for path in img_path:
	im = cv2.imread(path,0)
	im = np.float32(im)
	im = 255-im
	img.append(im/255)
img = nd.array(img)
output = net(img.reshape((7,1,28,28)))
print(nd.topk(output,k=3,ret_typ='indices'))
print(nd.argmax(output,axis=1))
