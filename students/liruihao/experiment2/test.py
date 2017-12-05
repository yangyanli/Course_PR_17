import itertools
import tensorflow as tf
import cv2
import numpy as np


def load(file_name):
	tva = []
	row_start_y = []
	row_end_y = []
	column_start_x = []
	column_end_x = []
	row_crop = []
	column_crop = []

	image = cv2.imread(file_name)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	rows = len(gray_image)
	columns = len(gray_image[0])
	rows_has_image = np.zeros(rows, bool)
	columns_has_image = np.zeros(columns, bool)
	rows_has_image[0] = False
	columns_has_image[0] = False

	for i in range(1, rows):
		for j in range(columns):
			if gray_image[i][j] < 200:
				rows_has_image[i] = True
				break
		if rows_has_image[i - 1] == False and rows_has_image[i] == True:
			temp = (int)(i - 1 - rows / 40)
			if temp < 0:
				temp = 0
			row_start_y.append(temp)
		elif rows_has_image[i - 1] == True and rows_has_image[i] == False:
			temp = (int)(i + rows / 40)
			if temp >= rows:
				temp = rows - 1
			row_end_y.append(temp)

	for i in range(len(row_start_y)):
		row_crop.append(gray_image[row_start_y[i]:row_end_y[i], 0:columns])

	for k in range(len(row_start_y)):
		for i in range(1, columns):
			for j in range(row_start_y[k], row_end_y[k]):
				if gray_image[j][i] < 200:
					columns_has_image[i] = True
					break
			if columns_has_image[i - 1] == False and columns_has_image[i] == True:
				temp = (int)(i - 1 - columns / 40)
				if temp < 0:
					temp = 0
				column_start_x.append(temp)
			elif columns_has_image[i - 1] == True and columns_has_image[i] == False:
				temp = (int)(i + columns / 40)
				if temp >= columns:
					temp = columns - 1
				column_end_x.append(temp)
		columns_has_image = np.zeros(columns, bool)
		columns_has_image[0] = False

	size_1 = int(len(column_start_x) / len(row_start_y))
	for i in range(len(row_start_y)):
		for j in range(size_1):
			width = column_end_x[j + i * size_1] - column_start_x[j + i * size_1]
			height = row_end_y[i] - row_start_y[i]
			if width < height:
				difference_1 = int((height - width) / 2)
				start_x = column_start_x[j + i * size_1] - difference_1
				if start_x < 0:
					start_x = 0
					difference_1 = column_start_x[j + i * size_1]

				difference_2 = int((height - width) / 2)
				end_x = column_end_x[j + i * size_1] + difference_2
				if end_x >= columns:
					end_x = columns - 1
					difference_2 = columns - 1 - column_end_x[j + i * size_1]

				temp_image = np.array(gray_image[row_start_y[i]:row_end_y[i], start_x:end_x])
				for s in range(len(temp_image)):
					for k in range(len(temp_image[s])):
						if k in range(0, difference_1) or k in range(end_x - start_x - difference_2, end_x - start_x):
							temp_image[s][k] = 255
				column_crop.append(temp_image)
			else:
				difference_1 = int((width - height) / 2)
				start_y = row_start_y[i] - difference_1
				if start_y < 0:
					start_y = 0
					difference_1 = row_start_y

				difference_2 = int((width - height) / 2)
				end_y = row_end_y[i] + difference_2
				if end_y >= rows:
					end_y = rows - 1
					difference_2 = rows - 1 - row_end_y[i]

				temp_image = np.array(gray_image[start_y:end_y, column_start_x[j + i * size_1]:column_end_x[j + i * size_1]])
				for s in range(len(temp_image)):
					if s in range(0, difference_1) or s in range(end_y - start_y - difference_2, end_y - start_y):
						for k in range(len(temp_image[s])):
							temp_image[s][k] = 255
				column_crop.append(temp_image)

	for i in range(len(column_crop)):
		column_crop[i] = cv2.resize(column_crop[i], (28, 28))
		tv = list(itertools.chain.from_iterable(column_crop[i]))
		tva.append([(255 - x) * 1.0 / 255.0 for x in tv])

	return tva


# def load(file_name):
# 	image = cv2.imread(file_name+".png")
# 	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	gray_image = cv2.resize(gray_image, (28,28))
# 	cv2.imwrite(file_name+"_gray.png", gray_image)
# 	tv = list(itertools.chain.from_iterable(gray_image))
# 	tva = [(255 - x) * 1.0 / 255.0 for x in tv]
# 	return tva


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

result = load("numbers_2.png")
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	init = tf.initialize_all_variables()
else:
	init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	saver.restore(sess, "my_net\\save_net.ckpt")  # 这里使用了之前保存的模型参数
	myprediction = tf.argmax(prediction, 1)
	predint = myprediction.eval(feed_dict={xs: result, keep_prob: 1.0}, session=sess)

	print('recognize result:')
	for i in range(len(predint)):
		print(predint[i])
