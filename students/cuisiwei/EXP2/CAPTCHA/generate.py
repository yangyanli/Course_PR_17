#coding=utf-8
#!/usr/bin/env python

#import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from captcha.audio import AudioCaptcha
from captcha.image import ImageCaptcha
import numpy as np
import random
import string
from PIL import Image
from attr import CaptchaAttr

num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lowercase = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

def getTraindata():
    image = ImageCaptcha(width = CaptchaAttr['imgWidth'],height=CaptchaAttr['imgHeight'])
    s = "".join([random.choice(num+UPPERCASE+lowercase) for len in range(CaptchaAttr['geneLen'])])
    data = image.generate(s)
    # print(data)
    #按灰度读入
    # arr = np.array(Image.open(data).convert('L')).flatten() / 255;
    arr = np.array(Image.open(data).convert('L'));
    #plt.imsave("./img/Train/"+s+".png", arr);
    image.write(s, "./img/Train/"+s+'.png')
    # print(arr.shape)
    return arr,s


if __name__ == '__main__':
    dir = "./CPATCHA/";
    time = 1;
    nxt_batch = 50;
    a,b  = getTraindata();
    # print(len(num+lowercase+UPPERCASE))
    print(a)
    #plt.imsave(b,a);
    #plt.imshow(a)
    #plt.show()


