from __future__ import print_function

import tensorflow as tf
import numpy as np
import struct
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet, dense_to_one_hot

# number 1 to 10 data
from tensorflow.python.framework import dtypes

def loadImageSet(filename):
	print("load image set", filename)
	binfile = open(filename, 'rb')
	buffers = binfile.read()

	head = struct.unpack_from('>IIII', buffers, 0)
	print("head,", head)

	offset = struct.calcsize('>IIII')
	imgNum = head[1]
	width = head[2]
	height = head[3]
	# [60000]*28*28

	imgs = np.frombuffer(buffers, dtype=np.uint8, offset=offset)
	binfile.close()

	imgs = imgs.reshape(imgNum, width, height, 1)
	print("imgs", imgs.shape[0])
	print("load imgs finished")
	return imgs


def loadLabelSet(filename):
	print("load label set", filename)
	binfile = open(filename, 'rb')
	buffers = binfile.read()

	head = struct.unpack_from('>II', buffers, 0)
	print("head,", head)

	offset = struct.calcsize('>II')
	print("offset,", offset)
	labels = np.frombuffer(buffers, dtype=np.uint8, offset=offset)
	binfile.close()

	labels = dense_to_one_hot(labels, 10)
	print("load label finished")
	return labels


def read_data_sets(fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):
	if fake_data:
		def fake():
			return DataSet(
					[], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

		train = fake()
		validation = fake()
		test = fake()
		return base.Datasets(train=train, validation=validation, test=test)

	TRAIN_IMAGES = "train-images.idx3-ubyte"
	TRAIN_LABELS = "train-labels.idx1-ubyte"
	TEST_IMAGES = "t10k-images.idx3-ubyte"
	TEST_LABELS = "t10k-labels.idx1-ubyte"

	train_images = loadImageSet(TRAIN_IMAGES)
	train_labels = loadLabelSet(TRAIN_LABELS)
	test_images = loadImageSet(TEST_IMAGES)
	test_labels = loadLabelSet(TEST_LABELS)

	if not 0 <= validation_size <= len(train_images):
		raise ValueError(
				'Validation size should be between 0 and {}. Received: {}.'
					.format(len(train_images), validation_size))

	validation_images = train_images[:validation_size]
	validation_labels = train_labels[:validation_size]
	train_images = train_images[validation_size:]
	train_labels = train_labels[validation_size:]

	options = dict(dtype=dtype, reshape=reshape, seed=seed)

	train = DataSet(train_images, train_labels, **options)
	validation = DataSet(validation_images, validation_labels, **options)
	test = DataSet(test_images, test_labels, **options)

	return base.Datasets(train=train, validation=validation, test=test)


mnist = read_data_sets(one_hot=True)


def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
	correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
	return result


def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1, name=name)
	return tf.Variable(initial)


def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape, name=name)
	return tf.Variable(initial)


def conv2d(x, W):
	# stride [1, x_movement, y_movement, 1]
	# Must have strides[0] = strides[3] = 1
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	# stride [1, x_movement, y_movement, 1]
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) / 255.  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32], name="W_conv1")  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32], name="b_conv1")
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64], name="b_conv2")
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W_fc1")
b_fc1 = bias_variable([1024], name="b_fc1")
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10], name="W_fc2")
b_fc2 = bias_variable([10], name="b_fc2")
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
saver = tf.train.Saver()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	init = tf.initialize_all_variables()
else:
	init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
	if i % 50 == 0:
		print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))

save_path = saver.save(sess, "my_net\\save_net.ckpt")
print("Save to path: ", save_path)
