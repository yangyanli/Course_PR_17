# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import sys
import numpy as np
import matplotlib.pyplot as plt


class predict():
	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape,stddev=0.1)
		return tf.Variable(initial)
	def bias_variable(self,shape):
		initial = tf.constant(0.1,shape = shape)
		return tf.Variable(initial)
	def conv2d(self,x,W):
		return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')
	def max_pool(self,x):
		return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
	def predict(self,a):

		
		model_path = sys.path[0] + '/mnist.ckpt'

		#mnist = read_data_sets("MNIST_data/", one_hot=True)
		x = tf.placeholder("float", shape=[None, 784])
		y_ = tf.placeholder("float", shape=[None, 10])


		filter1 = self.weight_variable([5,5,1,32])
		b_conv1 = self.bias_variable([32])

		x_image = tf.reshape(x,[-1,28,28,1])

		h_conv1 = tf.nn.relu(self.conv2d(x_image,filter1)+b_conv1)
		layer1_out = self.max_pool(h_conv1)

		filter2 = self.weight_variable([5,5,32,64])
		b_conv2 = self.bias_variable([64])
		h_conv2 = tf.nn.relu(self.conv2d(layer1_out,filter2)+b_conv2)
		layer2_out = self.max_pool(h_conv2)


		W_fc1 = self.weight_variable([7*7*64,1024])
		b_fc1 = self.bias_variable([1024])

		layer3_in = tf.reshape(layer2_out, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(layer3_in, W_fc1) + b_fc1)
		keep_prob = tf.placeholder("float")
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		W_fc2 = self.weight_variable([1024, 10])
		b_fc2 = self.bias_variable([10])

		y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

		cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		sess = tf.InteractiveSession()
		saver = tf.train.Saver()
		saver.restore(sess,model_path)
		print tf.argmax(y_conv,1).eval(feed_dict={x: a,keep_prob: 1.0})
