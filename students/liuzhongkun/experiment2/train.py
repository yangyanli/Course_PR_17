import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

row = 28
col = 28
input = row * col
class_num = 10
learning_rate = 0.001
batch_size = 50
acc_best = 0.0

mnist = input_data.read_data_sets("MNIST/", one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None,input])
y = tf.placeholder("float", shape=[None,class_num])
keep_prob = tf.placeholder("float")

def CNN(x,weight,bias, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.relu(tf.nn.conv2d(x, weight, strides=strides, padding=padding) + bias)
def MaxPool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(x, ksize=ksize, strides= strides, padding=padding)
def Ful_Con(x,weight,bias):
    return tf.nn.relu(tf.matmul(x,weight) + bias)


x1 = tf.reshape(x,[-1,28,28,1])
weight_cnn1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
bias_cnn1 = tf.Variable(tf.constant(0.1, shape = [32]))
x2 = CNN(x1,weight_cnn1,bias_cnn1)
x3 = MaxPool(x2)
'''
weight_cnn2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
bias_cnn2 = tf.Variable(tf.constant(0.1, shape = [64]))
x4 = CNN(x3,weight_cnn2,bias_cnn2)
x5 = MaxPool(x4)
'''
x6 = tf.reshape(x3,[-1,32*14*14])
weight_fulcon1 = tf.Variable(tf.truncated_normal([14*14*32,1024], stddev=0.1))
bias_fulcon1 = tf.Variable(tf.truncated_normal([1024], stddev=0.1))
x7 = Ful_Con(x6,weight_fulcon1,bias_fulcon1)
x8 = tf.nn.dropout(x7, keep_prob)
weight_fulcon2 = tf.Variable(tf.truncated_normal([1024, class_num],stddev=0.1))
bias_fulcon2 = tf.Variable(tf.truncated_normal([class_num], stddev=0.1))

y1 = tf.nn.softmax(tf.matmul(x8, weight_fulcon2) + bias_fulcon2)

cross_entropy = -tf.reduce_sum(y*tf.log(y1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
correct_prediction = tf.cast(tf.equal(tf.argmax(y1,1), tf.argmax(y,1)),dtype = "float")

accuracy = tf.reduce_mean(correct_prediction)
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
sess.run(tf.initialize_all_variables())
for epoch in range(2500):
    batch = mnist.train.next_batch(100)
    sess.run(optimizer, feed_dict={
        x: batch[0],
        y: batch[1],
        keep_prob:0.7
    })
    if epoch%100 == 0:
        acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Accuracy:", acc)
        if acc > acc_best:
            saver_path = saver.save(sess, "save/one_model.ckpt")
            acc_best = acc
