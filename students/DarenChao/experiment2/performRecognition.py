import cv2
import tensorflow as tf
import sys

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def restore_sigmiod():
    # load the classifier
    w = tf.Variable(tf.random_normal([784, 10], stddev=0.01))
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./Model_sigmoid/sigmoid_model.ckpt")
    X = tf.placeholder("float", [None, 784])
    py_x = tf.matmul(X, w)
    predict_op = tf.argmax(py_x, 1)
    return sess, predict_op, X

def getResult_sigmoid(roi, sess, predict_op, X):
    nbr = sess.run(predict_op, feed_dict={X: [roi]})
    return nbr

def restore_cnn():
    # init cnn
    X = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(X, [-1, 28, 28, 1])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_pre = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    prediction = tf.argmax(y_pre, 1)
    # load the classifier
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, "./Model_cnn/cnn_model.ckpt")
    return sess, prediction, X, keep_prob

def getResult_cnn(roi, sess, prediction, X, keep_prob):
    labelPre = sess.run(prediction, feed_dict={X: [roi], keep_prob:1})
    return labelPre

if __name__ == '__main__':

    # Get the path of the training set
    # parser = ap.ArgumentParser()
    # parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", required="True")
    # parser.add_argument("-i", "--image", help="Path to Image", required="True")
    # args = vars(parser.parse_args())
    args = {}
    args["classiferPath"] = './digits_cls.pkl'
    args["image"] = './photo_cdr2.jpg'
    image = cv2.imread(args["image"])

    # Convert to grayscale and apply Gaussian filtering
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # Threshold the image
    # cv2.threshold(src,thresh,maxval,type[,dst])->retval,dst
    # thresh:阈值，maxval:在二元阈值THRESH_BINARY和逆二元阈值THRESH_BINARY_INV中使用的最大值
    # 返回值retval其实就是阈值 type:使用的阈值类型
    # ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV) # photo_1 and photo_2
    _, image_threshold = cv2.threshold(image_gray, 124, 255, cv2.THRESH_BINARY_INV) # photo_cdr1 and photo_cdr2
    # ret, im_th = cv2.threshold(im_gray, 110, 255, cv2.THRESH_BINARY_INV) # photo_3 and photo_4

    # Find contours in the image
    # 第二个参数表示轮廓的检索模式，cv2.RETR_EXTERNAL表示只检测外轮廓
    # 第三个参数method为轮廓的近似办法，cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标
    # 返回三个值，一个是图像本身，一个是轮廓集合，还有一个是每条轮廓对应的属性
    _, contours, __ = cv2.findContours(image_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    # 得到包含轮廓的矩形，返回值是顶点和宽高
    rectangles = [cv2.boundingRect(contour) for contour in contours]

    # Get sigmoid or cnn parameters
    # sigmoid_parameters = restore_sigmiod()
    cnn_parameters = restore_cnn()

    # For each rectangular region, calculate HOG features and predict
    for rect in rectangles:
        # Draw the rectangles | cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        x = rect[0]; y = rect[1]; w = rect[2]; h = rect[3]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 133, 133), 1)
        # Make the rectangular region around the digit, get region of interest(ROI)
        new_h = int(h * 1.618)
        new_y = int(y + h // 2 - new_h // 2)
        new_x = int(x + w // 2 - new_h // 2)
        ROI = image_threshold[new_y:new_y + new_h, new_x:new_x + new_h]
        # Resize the image
        try:
            ROI = cv2.resize(ROI, (28, 28), interpolation=cv2.INTER_AREA) # 区域插值
        except:
            print("Wrong contour!")
            exit(-1)
        ROI = cv2.dilate(ROI, (3, 3)) # 膨胀
        ROI = ROI.flatten() # 变平
        ROI = ROI / 255

        # Calculate the HOG features
        labelPre = getResult_cnn(ROI, cnn_parameters[0], cnn_parameters[1], cnn_parameters[2], cnn_parameters[3])
        # nbr = getResult_sigmoid(roi, sigmoid_parameters[0], sigmoid_parameters[1], sigmoid_parameters[2])
        print(labelPre, end=' ')
        cv2.putText(image, str(int(labelPre[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (199, 0, 0), 1)

    cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
    cv2.imshow("Resulting Image with Rectangular ROIs", image)
    cv2.waitKey()
