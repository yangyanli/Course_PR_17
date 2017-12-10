from preproforwritten import *
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter, ImageOps

def padding(image, each_domain, height = 28, width = 28):
    im_o = Image.fromarray(image)
    output = []
    each_domain_new = sorted(each_domain, key=lambda a: a[1])
    each_domain_new = sorted(each_domain_new, key=lambda a: a[0])
    for domain in each_domain_new:
        x = domain[2] - domain[0] + 1
        y = domain[3] - domain[1] + 1
        x1 = domain[0]
        x2 = domain[1]
        x3 = domain[2] + 1
        x4 = domain[3] + 1

        length = max([x, y]) + 2
        real_pic = np.ones([length,length]) * 255
        real_pic[int(length/2 - x/2):int(length/2 + x/2),int(length/2 - y/2):int(length/2 + y/2)] = image[x1:x3,x2:x4]

        for i in range(len(real_pic)):
            for j in range(len(real_pic[0])):
                real_pic[i][j] = 1.0 - float(real_pic[i][j]) / 255.0


        im = Image.fromarray(real_pic)
        im_r = im.resize([20,20])
        real_p = np.array(im_r.getdata())
        real_p = real_p.reshape([20,20])
        tmp = np.zeros([28,28])
        tmp[4:24,4:24] = real_p
        output.append(tmp)
        '''
        im = Image.fromarray(real_pic)
        im_r = im.resize([28, 28])
        real_p = np.array(im_r.getdata())
        real_p = real_p.reshape([28, 28])
        output.append(real_p)
        '''
    return output


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    img = Image.open("./test/ex5.jpg")

    im = np.array(img)
    tmp = np.zeros([len(im),len(im[0])])
    for i in range(len(im)):
        for j in range(len(im[0])):
            if max(im[i][j]) > 80:
                tmp[i][j] = 255
    tmp = tmp[:,6:]
    # img = Image.fromarray(tmp)
    # img.show()
    # exit(0)

    num, each_domain, image_out =  image_get(tmp)
    image = padding(image_out, each_domain)
    for i in range(len(image)):
        tmp = Image.fromarray(image[i])
        tmp = tmp.filter(ImageFilter.MaxFilter(3))
        image[i] = np.array(tmp)
    img = img.filter(ImageFilter.MinFilter(3))
    '''
    tmp = []
    for i in range(10):
        tmp = tmp + list(image[i]*255)
    a = np.array(tmp)
    b = Image.fromarray(a)
    b.show()
    exit(0)
    '''
    '''
    for i in range(28):
        for j in range(28):
            if image[0][i][j] < 64:
                plt.plot(j, -i,'r.')
    plt.show()
    '''
    # exit(0)
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", shape=[None, 784])
    # y_ = tf.placeholder("float", shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    '''
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    '''
    W_fc1 = weight_variable([14 * 14 * 32, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    # train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    result = tf.argmax(y_conv, 1)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    saver = tf.train.Saver()
    saver.restore(sess, 'save/model.ckpt')
    result_list=[]
    image_list = []
    for i in range(len(each_domain)):
        tmp = result.eval(feed_dict={x : np.reshape(image[i], [1,784]), keep_prob : 1.0})
        result_list.append(tmp)
        image_list.append(image[i])

    i = 0
    counter = 1
    while(i < len(result_list)):
        counter += 1
        a = result_list[i:min(i+10, len(result_list))]
        a = [str(p) for p in a]
        print(' '.join(a))
        i += 10
    tmp = []
    '''
    im_all = np.array([[0 for i in range(28*10)] for j in range(28*counter)])
    i = 0
    j = 0
    for i in range(len(image_list)):
        if j > 0 and j%10 == 0:
            i += 1
            j = 0
        im_all[(i*28) : ((i + 1)*28),(j*28) : ((j + 1)*28)] = np.array(image_list[i] * 255)
    im = Image.fromarray(im_all)
    im.show()
    exit(0)
    '''
    for i in range(len(image_list)):
        if i > 0 and i%10 == 0:
            im = Image.fromarray(np.array(tmp))
            im.show()
            tmp = []
        tmp = tmp + list(image_list[i] * 255)
    if tmp != [] :
        im = Image.fromarray(np.array(tmp))
        im.show()
