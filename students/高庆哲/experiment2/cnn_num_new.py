import tensorflow as tf
import numpy as np
from PIL import Image
import os
class dataset:
    def __init__(self, data,label):
        self.data = data
        self.label = label
        self.index = 0

    def next_set(self, num):
        self.index += num
        if(self.index>len(self.data)):
            self.clear_index()
            self.index += num
        return self.data[(self.index - num):self.index],self.label[(self.index - num):self.index]

    def clear_index(self):
        self.index = 0

    def all_data(self):
        return self.data,self.label

    def next_batch(self,num):
        '''
        Return a total of `num` random samples and labels.
        '''
        idx = np.arange(0, len(self.data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [self.data[i] for i in idx]
        labels_shuffle = [self.label[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def trans_num_index(num):
    return num-1

def trans_letter_index(letter):
    return ord(letter)-88

def dealdata(model,num):

    data = []
    for i in range(num):
        pic = Image.open('genpic/'+model+'/'+str(i)+'.png')

        pic = pic.convert("1")



        a = np.reshape(np.array(pic), [9600])

        a = [int(i) for i in a ]

        data.append(a)

        if i%100==0:
            print(i)
    data = np.array(data)

    np.save('genpic/'+model+'_data'+'.npy',data)

def getdata(model):
    label = np.load('genpic/'+model+'_label'+'.npy')
    data =  np.load('genpic/'+model+'_data'+'.npy')

    return dataset(data,label)


def train(size,model):


    data = getdata(model)
    x = tf.placeholder(tf.float32,shape=[None,160*60])
    y_ = tf.placeholder(tf.float32,shape=[None,10*size])



    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],  padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


    def new_con(x,W):
        return tf.nn.conv2d(x, W, strides=[1, 20, 7, 1], padding='VALID')


    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 160, 60, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([20, 8, 64, 10])
    b_conv3 = bias_variable([10])
    h_conv3 = tf.nn.relu(new_con(h_pool2, W_conv3) + b_conv3)

    h_conv3 = tf.reshape(h_conv3,shape = [-1,4,10])

    y_conv = tf.transpose(h_conv3,perm=[0,2,1])
    y_conv = tf.reshape(y_conv,[-1,4*10])



    # h_pool2_flat = tf.reshape(h_pool2, [-1, 40 * 15 * 64])
    # W_fc1 = weight_variable([40 * 15 * 64, 1024])
    # b_fc1 = bias_variable([1024])
    #
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #
    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #
    # W_fc2 = weight_variable([1024, 10*size])
    # b_fc2 = bias_variable([10*size])
    #
    # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # cross_entropy = tf.reduce_mean(
    #     -tf.reduce_sum(y_ [0:10]* tf.log(tf.clip_by_value(y_conv[0:10], 1e-10, 1.0))))


    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y_conv))
    # for onekey in range(1,size):
    #     cross_entropy += tf.reduce_mean(
    #         -tf.reduce_sum(y_ [size*onekey:size*onekey+10]* tf.log(tf.clip_by_value(y_conv[size*onekey:size*onekey+10], 1e-10, 1.0))))

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.96, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv[0:10], 1), tf.argmax(y_[0:10], 1))




    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for onekey in range(1, size):
        correct_prediction = tf.equal(tf.argmax(y_conv[size*onekey:size*onekey+10], 1), tf.argmax(y_[size*onekey:size*onekey+10], 1))
        accuracy += tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy /=size


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(2000000):
            part_data,part_label = data.next_batch(50)
            train_step.run(feed_dict={x: part_data, y_: part_label})

            if i%100==0:
                file = open(model+'cnn_train_info.txt', 'a')
                train_accuracy = accuracy.eval(feed_dict={
                    x: part_data, y_: part_label})



                # print('W_conv1',sess.run(W_conv1))
                # print('W_conv2',sess.run(W_conv2))
                # print('W_fc2',sess.run(W_fc2))
                # print('W_fc1',sess.run(W_fc1))


                # print('y_conv', sess.run(y_conv, feed_dict={
                #     x: part_data, y_: part_label, keep_prob: 1.0}))
                print('step %d, training accuracy %g' % (i, train_accuracy))
                print('loss', sess.run(cross_entropy, feed_dict={
                    x: part_data, y_: part_label,}))
                file.write('step %d, training accuracy %g' % (i, train_accuracy))
                file.write('\n')

                file.write('loss'+str(sess.run(cross_entropy, feed_dict={
                    x: part_data, y_: part_label}))+'\n')

                file.close()

            if i%10000==0 and i!=0:
                saver = tf.train.Saver()
                os.mkdir('newmodel/+'+model+str(i/10000)+'w')
                saver.save(sess, 'newmodel/'+model+str(i/10000)+'/model')

def load(model_path,size,model):

    data = getdata(model)
    x = tf.placeholder(tf.float32,shape=[None,160*60])
    y_ = tf.placeholder(tf.float32,shape=[None,10*size])



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

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 160, 60, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 40 * 15 * 64])

    W_fc1 = weight_variable([40 * 15 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10*size])
    b_fc2 = bias_variable([10*size])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ [0:10]* tf.log(tf.clip_by_value(y_conv[0:10], 1e-10, 1.0))))


    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y_conv))
    # for onekey in range(1,size):
    #     cross_entropy += tf.reduce_mean(
    #         -tf.reduce_sum(y_ [size*onekey:size*onekey+10]* tf.log(tf.clip_by_value(y_conv[size*onekey:size*onekey+10], 1e-10, 1.0))))

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.96, staircase=True)

    # correct_prediction = tf.equal(tf.argmax(y_conv[0:10], 1), tf.argmax(y_[0:10], 1))
    #
    #
    #
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # for onekey in range(1, size):
    #     correct_prediction = tf.equal(tf.argmax(y_conv[size*onekey:size*onekey+10], 1), tf.argmax(y_[size*onekey:size*onekey+10], 1))
    #     accuracy += tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    # accuracy /=size

    predict = tf.reshape(y_conv, [-1, 4, 10])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y_, [-1, 4, 10]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())



        saver.restore(sess, model_path)

        print("Model restored from file: %s" % model_path)

        file = open('result_num_8wnew.txt','a')
        all = 0
        for i in range(0,200):
            all_data, all_label = data.next_set(100)

            result = accuracy.eval(feed_dict={
                x: all_data, y_: all_label, keep_prob: 1.0})
            all +=result

            print('test accuracy %g' % result )

            file.write('test accuracy %g' % result +'\n')

        all  = all/200

        file.write('all average test accuracy %g' % all + '\n')

    print('0000')

pass

if __name__ == '__main__':
    # dealdata('train_num',20000)
    # dealdata('test_num', 20000)
    train(4,'train_num')
    # load( 'model/train_num8.0/model',4,'test_num')
    pass

