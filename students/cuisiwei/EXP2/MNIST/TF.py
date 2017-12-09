from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

if __name__ == '__main__':
#原始数据
    mnist=input_data.read_data_sets('./MNIST/',one_hot=True)
#x是一个占位符，每一张图展平成784维的向量。用2维的浮点数张量来表示这些图
# 张量的形状是[None，784 ]。（None表示任何长度）
    x=tf.placeholder("float",[None,784])
#Variable代表一个可修改的张量

    W=tf.Variable(tf.zeros([784,10]))
    b=tf.Variable(tf.zeros([10]))
    y=tf.matmul(x,W)+b
#为计算交叉熵，需要添加一个新的占位符用于输入正确值
    y_=tf.placeholder("float",[None,10])
#回归模型
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#Softmax回归 计算交叉熵
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
#梯度下降算法
    # train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#初始化变量
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    # sess=tf.InteractiveSession()
    # tf.global_variables_initializer().run()
    for _ in range(5000):
        #随机抓取训练数据中的100个数据点
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    #tf.argmax:给出某个tensor对象在某一维上的其数据最大值所在的索引值
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    #正确率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,
      y_:mnist.test.labels}))

