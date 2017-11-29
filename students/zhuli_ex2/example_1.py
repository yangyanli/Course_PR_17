import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("F:\Git\Course_PR_17\experiment1\data\MNIST",one_hot=True)

sess=tf.InteractiveSession()


def weight(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def baise(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#convolution
#SAME代表卷积后图像大小不变
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#pooling
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def print_matrix(board):
    shapex=np.shape(board)
    batch=shapex[0]
    x=shapex[1]
    y=shapex[2]
    channel=shapex[3]

    for i in range(batch):
        tmp=np.zeros(shape=(x,y))
        for j in range(x):
          for k in range(y):
            tmp[j][k]=board[i][j][k][0]

        for row in tmp:
          print(row)
          print()

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])

Weight_1=weight([3,3,1,32])
baise_1=baise([32])
conv1=tf.nn.relu(conv2d(x_image,Weight_1)+baise_1)
pool1=max_pool(conv1)

Weight_2=weight([3,3,32,64])
baise_2=baise([64])
conv2=tf.nn.relu(conv2d(pool1,Weight_2)+baise_2)
pool2=max_pool(conv2)

#fc1,将2次池化后的特征图转化为1D向量，全连接层
W1=weight([7*7*64,1024])
b1=baise([1024])
h_p2_flat=tf.reshape(pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_p2_flat,W1)+b1) #运算、激活

#dropout
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob) #dropout操作

W2=weight([1024,10])
b2=baise([10])

#softmax分类器（分成10类，用概率表示），output为输出
output=tf.nn.softmax(tf.matmul(h_fc1_drop,W2)+b2)

cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(output),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.005).minimize(cost)
correct_prediction=tf.equal(tf.argmax(output,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()
saver=tf.train.Saver()
saver.restore(sess,"F:/Git/Course_PR_17/experiment2/saver/save.chpt")

# for i in range(1, 2001):
#     batch=mnist.train.next_batch(50)
#     train_step.run(feed_dict={x:batch[0],y:batch[1],keep_prob:0.7})
#
#     if i%50==0:
#         train_accuracy=accuracy.eval(feed_dict={x:batch[0],y:batch[1],keep_prob:1.0})
#         test_accuracy=accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
#         print("step: %d  train accuracy: %.3f test accuracy: %.3f"%(i,train_accuracy,test_accuracy))
#     if i%200==0:
#         saver.save(sess,"F:/Git/Course_PR_17/experiment2/saver/save.chpt")

#saver.save(sess,save_path='F:\Git\Course_PR_17\experiment2',global_step=1)
#for line in open("F:\Git\Course_PR_17\experiment2/data.txt",'rb'):
 #   print(line)

#图片处理
image=Image.open('F:\Git\Course_PR_17\experiment2\img\pic6.png')
image=image.convert('L')

#转化为二值图
WHITE,BLACK=255,0
image=image.point(lambda P:BLACK if P>100 else WHITE)
image=image.convert('1')

#切割图片
inletter=False
foundletter=False
start=0
end=0

letters2=[]
for yy in range(image.size[1]):
    for xx in range(image.size[0]):
        pix=image.getpixel((xx,yy))
        if pix!=0:
            inletter=True

    if foundletter==False and inletter==True:
        foundletter=True
        start=yy

    if foundletter==True and inletter==False:
        foundletter=False
        end=yy-1
        letters2.append((start,end))

    inletter=False

letters=[]
for i in range(letters2.__len__()):
    for xx in range(image.size[0]):
        for yy in range(letters2[i][0],letters2[i][1]):
            pix=image.getpixel((xx,yy))
            if pix!=0:
                inletter=True

        if foundletter==False and inletter==True:
            foundletter=True
            start=xx

        if foundletter==True and inletter==False:
            foundletter=False
            end=xx-1
            letters.append((start,end))

        inletter=False

print(letters)
print(letters2)

l1=letters.__len__()
l2=letters2.__len__()


k=1;
j=0
mat_image=np.zeros((l1,784))
for i in range(l1):
    if i>1 and letters[i][0]<letters[i-1][1] and j<l2:
        j=j+1
# for j in range(l2):
#     mat_image=np.zeros((l1,784))
#     labels = []
#     for i in range(l1):
    region = (letters[i][0]-10, letters2[j][0]-5, letters[i][1]+10, letters2[j][1]+5)
    im=image.crop(region)
    m1=im.resize((28,28))

    for a in range(28):
        for b in range(28):
            pix = m1.getpixel((b, a))
            if pix == 255:
                mat_image[i][a * 28 + b] = 1
            else:
                mat_image[i][a * 28 + b] = 0

res=output.eval(feed_dict={x:mat_image,keep_prob:1.0})

for r in res:
    max_p = 0.0
    lab = -1
    for i in range(10):
        print('The probability of '+str(i)+ ': '+ str(r[i]))
        if r[i] > max_p:
            max_p = r[i]
            lab = i
    print('The result: '+str(lab))

image.show()