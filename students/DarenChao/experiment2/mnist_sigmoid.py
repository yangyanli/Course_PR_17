from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # 获取mnist 数据
    mnist = input_data.read_data_sets("./Mnist_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    # 定义占位符变量，X维度784， Y维度是10
    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])

    w = tf.Variable(tf.random_normal([784, 10], stddev=0.01))
    py_x = tf.matmul(X, w)

    # 定义损失函数，交叉熵损失函数 y=sigmoid(X∗W)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

    # 训练操作，梯度下降，最小化损失函数
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    # train_op = tf.train.AdamOptimizer(0.5).minimize(cost)

    # 预测操作，按行取最大值
    predict_op = tf.argmax(py_x, 1)

    # 保存模型
    saver = tf.train.Saver()

    # 定义会话
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # 调用多次梯度下降
    for i in range(100):
        # 训练，每个batch的大小是128
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        # for j in range(0, len(trX)):
        #     sess.run(train_op, feed_dict={X: [trX[j]], Y: [trY[j]]})
        # 测试所有的测试数据，并输出效果的平均值
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY})))

    saver.save(sess, "./Model_sigmoid/sigmoid_model.ckpt")