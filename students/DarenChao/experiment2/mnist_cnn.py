from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def compute_accuracy(v_x, v_y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={X:v_x, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy,feed_dict={X: v_x, Y: v_y, keep_prob:1})
    return result

def getConv(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

if __name__ == '__main__':
    mnist = input_data.read_data_sets("./Mnist_data/", one_hot=True)

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(X, [-1, 28, 28, 1])

    # ********************** conv1 *********************************
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(getConv(x_image, W_conv1) + b_conv1)  # output size 28*28*32
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # output size 14*14*32

    # ********************** conv2 *********************************
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(getConv(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # output size 7*7*64

    # ********************* func1 layer *********************************
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # ********************* func2 layer *********************************
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # init saver
    saver = tf.train.Saver()
    # init session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    training_iters = 2000
    batch_size = 128
    for i in range(training_iters):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
        if i % 50 == 0:
            print(i, compute_accuracy(mnist.test.images, mnist.test.labels))

    saver.save(sess, "./Model_cnn/cnn_model.ckpt")
