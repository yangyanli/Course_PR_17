#coding=utf-8
#!/usr/bin/env python
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from captcha.audio import AudioCaptcha
from captcha.image import ImageCaptcha
import numpy as np
import math
import random
import tempfile
import string
from PIL import Image

from attr import CaptchaAttr
from generate import num
from generate import lowercase
from generate import UPPERCASE



def getTraindata():
    image = ImageCaptcha(width = CaptchaAttr['imgWidth'],height=CaptchaAttr['imgHeight'])
    s = "".join([random.choice(num+UPPERCASE+lowercase) for len in range(CaptchaAttr['geneLen'])])
    data = image.generate(s)

    # print(data)
    #按灰度读入
    # arr = np.array(Image.open(data).convert('L')).flatten() / 255;
    arr1 = np.array(Image.open(data));
    arr = np.array(Image.open(data).convert('L'));

    # plt.imsave("./img/Train/"+s+".png", arr1);
    # image.write(s, "./img/Train/"+s+'.png')
    # print(arr.shape)
    return arr,s,arr1


def nxt_batch(size = CaptchaAttr['BatchSize']):
    data = np.zeros((size,CaptchaAttr['imgHeight']*CaptchaAttr['imgWidth']));
    label = np.zeros((size,CaptchaAttr['geneLen']*CaptchaAttr['sourceLen']))
    for i in range(size):
        timg, tlabel,img = getTraindata()
        if vis:
            print("label:"+tlabel)
        # plt
        data[i, :] = timg.flatten()*1.0 / 255.0
        tlabel2 = translateText(tlabel)
        label[i] = tlabel2;

    return data, label,img

def getText(vec):
    ch = num+UPPERCASE+lowercase;
    text = "";
    for i in range(CaptchaAttr['geneLen']):
        text += ch[vec[i]];
    # print(text)
    return text

def translateText(text):
    vec = np.zeros((CaptchaAttr['geneLen']*CaptchaAttr['sourceLen']))
    for i in range(CaptchaAttr['geneLen']):
        ch = ord(text[i]) - ord('0');
        if ch>9:
            ch -= (ord('A')-ord('9')-1);
            if ch>35:
                ch -= (ord('a')-ord('Z')-1);
        vec[i * CaptchaAttr['sourceLen'] + ch] = 1
    return vec
vis = True
if __name__ == '__main__':
    #原始数据
    # dir = "./MNIST/";

    # mnist = input_data.read_data_sets(dir, one_hot=True);

    #占位符
    x = tf.placeholder(tf.float32,[None,CaptchaAttr['imgHeight']*CaptchaAttr['imgWidth']])
    y_ = tf.placeholder(tf.float32,[None,CaptchaAttr['geneLen']*CaptchaAttr['sourceLen']])



    sz = 3
    with tf.name_scope('reshape'):
        x_img = tf.reshape(x,[-1,CaptchaAttr['imgHeight'],CaptchaAttr['imgWidth'],1]);
    with tf.name_scope("conv1"):
        layer_name = "layerconv1";
        W_conv1 = tf.Variable(tf.truncated_normal([sz, sz, 1, 32], stddev=0.1));
        tf.summary.histogram(layer_name + "/weights", W_conv1)  # 可视化观看变量
        b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]));
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_img,W_conv1,strides=[1,1,1,1],padding="SAME")+b_conv1);
    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('conv2'):
        layer_name = "layerconv2";

        W_conv2 = tf.Variable(tf.truncated_normal([sz,sz,32,64],stddev=0.1));
        tf.summary.histogram(layer_name + "/weights", W_conv2)  # 可视化观看变量

        b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]));
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding="SAME")+b_conv2);
    with tf.name_scope('pool2'):
        h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.name_scope('conv3'):
        layer_name = "layerconv3";
        W_conv3 = tf.Variable(tf.truncated_normal([sz, sz, 64, 64], stddev=0.1));
        tf.summary.histogram(layer_name + "/weights", W_conv3)  # 可视化观看变量
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]));
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3);
    with tf.name_scope('pool3'):
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('fc1'):
        layer_name = "layerfc1";
        #W_fc1 = tf.Variable(tf.truncated_normal([15 * 40 * 128, 1024],stddev=0.1));
        W_fc1 = tf.Variable(tf.truncated_normal([int(math.ceil(CaptchaAttr['imgWidth']/2.0/2.0/2.0)) * int(math.ceil(CaptchaAttr['imgHeight']/2.0/2.0/2.0)) * 64, 1024],stddev=0.1));

        tf.summary.histogram(layer_name + "/biases", W_fc1)  # 可视化观看变量
        b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]));
        h_pool3_flat = tf.reshape(h_pool3,[-1,int(math.ceil(CaptchaAttr['imgWidth']/2.0/2.0/2.0)) * int(math.ceil(CaptchaAttr['imgHeight']/2.0/2.0/2.0)) * 64]);

        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1);
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32);
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob);
    with tf.name_scope('fc2'):
        layer_name = "layerfc2";
        W_fc2 = tf.Variable(tf.truncated_normal([1024,CaptchaAttr['geneLen']*CaptchaAttr['sourceLen']],stddev=0.1));
        tf.summary.histogram(layer_name + "/biases", W_fc2)  # 可视化观看变量
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[CaptchaAttr['geneLen']*CaptchaAttr['sourceLen']]));
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2;
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y_conv);
        cross_entropy = tf.reduce_mean(cross_entropy);
        tf.summary.scalar('loss', cross_entropy)  # 可视化观看常量
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(CaptchaAttr['LearningRate']).minimize(cross_entropy);
    with tf.name_scope('accuracy'):
        y_trans = tf.reshape(y_, [-1, CaptchaAttr['geneLen'], CaptchaAttr['sourceLen']])
        y_convtrans = tf.reshape(y_conv, [-1, CaptchaAttr['geneLen'], CaptchaAttr['sourceLen']])
        y_trans2 = tf.argmax(y_trans, 2)
        y_convtrans2 = tf.argmax(y_convtrans, 2)
        correct_pred = tf.equal(y_trans2, y_convtrans2)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver();
    # config_gpu = tf.ConfigProto()
    config_gpu = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=config_gpu)) as sess:

    # config_gpu.gpu_options.allow_growth = True
    # with tf.Session(config=config_gpu) as sess:
    # with tf.Session() as sess:

        #sess.run(tf.global_variables_initializer());
        saver.restore(sess,'./CAPTCHA/model.ckpt')
        # saver.restore(sess,"./CAPTCHA/model.ckpt");
        for i in range(500):
            img,label,im = nxt_batch(1);
            # print("label:"+str(label))
            train_accu = accuracy.eval(feed_dict={
                x: img, y_: label, keep_prob: 1.0
            })
            if vis:
                print("res  :"+getText(y_convtrans2.eval(feed_dict={x: img, keep_prob: 1.0})[0]))
                plt.imshow(im)
                plt.show()

        print('Times: %d Accuracy %g' % (i,  train_accu))

