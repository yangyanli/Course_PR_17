import tensorflow as tf
import numpy as np
import PIL
from captcha.image import ImageCaptcha, ImageFilter
from PIL.Image import Image
from PIL.ImageDraw import Draw
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class WeakImageCaptcha(ImageCaptcha):
    """
    Override ImageCaptcha class in captcha library to generate custom captcha
    """
    table = []
    for i in range(256):
        table.append(i * 1.97)

    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = PIL.Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = PIL.Image.new('RGBA', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), PIL.Image.BILINEAR, expand=1)

            # warp
            #dx = w * random.uniform(0.1, 0.3)
            dx = w * 0.1
            #dy = h * random.uniform(0.2, 0.3)
            dy = h * 0.1
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), PIL.Image.QUAD, data)
            return im

        images = []
        for c in chars:
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        for im in images:
            w, h = im.size
            mask = im.convert('L').point(self.table)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image

    def generate_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = random_color(238, 255)
        color = random_color(0, 200, random.randint(220, 255))
        im = self.create_captcha_image(chars, color, background)
        self.create_noise_dots(im, color)
        self.create_noise_curve(im, color)
        im = im.filter(ImageFilter.SMOOTH)
        return im

def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)


# Utility functions from TensorFlow official tutorial
def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 5, 5, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


class CaptchaRecognizer:

    charset = "0123456789"
    DIGIT_IMAGE_WIDTH = 36
    DIGIT_IMAGE_HEIGHT = 60
    LABEL_NUM = len(charset)

    def __init__(self):
        self.index_map = {}
        for i in range(len(self.charset)):
            self.index_map[self.charset[i]] = i

        self.captcha_generator = WeakImageCaptcha()
        """
        example_chapcha, example_label = self.generate_captcha_sample(1)
        plt.imshow(np.asarray(example_chapcha[0]))
        plt.show()
        splited_example = self.split_character_4(example_chapcha[0])
        for s in splited_example:
            plt.imshow(np.asarray(s))
            plt.show()
        """

        self.build_nn()
        self.tf_saver = tf.train.Saver()
        self.tf_sess = tf.Session()
        self.tf_sess.run(tf.global_variables_initializer())

    def __del__(self):
        if self.tf_sess is not None:
            self.tf_sess.close()

    def build_nn(self):
        with tf.name_scope("input"):
            self.tf_image = tf.placeholder(tf.float32, [None, self.DIGIT_IMAGE_HEIGHT, self.DIGIT_IMAGE_WIDTH], name="image")
            self.tf_label = tf.placeholder(tf.float32, [None, self.LABEL_NUM], name="label")

            x_image = tf.reshape(self.tf_image, [-1, self.DIGIT_IMAGE_HEIGHT, self.DIGIT_IMAGE_WIDTH, 1])
          #  tf.summary.image('input', x_image, max_outputs=self.LABEL_NUM)

        with tf.name_scope("layer1"):
            W_conv1 = weight_variable([7, 7, 1, 32], name="W_conv1")
            b_conv1 = bias_variable([32], name="b_conv1")

            h_conv1 = tf.nn.tanh(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1,
                                     ksize=[1, 5, 5, 1],
                                     strides=[1, 2, 1, 1],
                                     padding='SAME')

        with tf.name_scope("layer2"):
            W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")
            b_conv2 = bias_variable([64], name="b_conv2")

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = tf.nn.max_pool(h_conv2,
                                     ksize=[1, 5, 5, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')

        with tf.name_scope("layer3"):
            W_conv3 = weight_variable([5, 5, 64, 128], name="W_conv3")
            b_conv3 = bias_variable([128], name="b_conv3")

            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = tf.nn.max_pool(h_conv2,
                                     ksize=[1, 5, 5, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')

        with tf.name_scope('densely-connected'):
            W_fc1 = weight_variable([self.DIGIT_IMAGE_WIDTH * self.DIGIT_IMAGE_HEIGHT * 8, 1024], name="W_fc1")
            b_fc1 = bias_variable([1024], name="b_fc1")

            h_pool3_flat = tf.reshape(h_pool3, [-1, self.DIGIT_IMAGE_WIDTH * self.DIGIT_IMAGE_HEIGHT * 8])
            h_fc1 = tf.nn.tanh(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        with tf.name_scope('dropout'):
            self.tf_keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            h_fc1_drop = tf.nn.dropout(h_fc1, self.tf_keep_prob)

        with tf.name_scope('readout'):
            W_fc2 = weight_variable([1024, self.LABEL_NUM])
            b_fc2 = bias_variable([self.LABEL_NUM])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        with tf.name_scope('loss'):
            self.tf_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_label, logits=y_conv))
            self.tf_train_step = tf.train.AdamOptimizer(1e-4).minimize(self.tf_cross_entropy)

        self.tf_predict = tf.argmax(y_conv, axis=1)
        self.tf_expect = tf.argmax(self.tf_label, axis=1)

        with tf.name_scope('evaluate_accuracy'):
            self.tf_correct_prediction = tf.equal(self.tf_predict, self.tf_expect)
            self.tf_accuracy = tf.reduce_mean(tf.cast(self.tf_correct_prediction, tf.float32))


    def train_with_generated_data(self):

        for i in range(1, 20001):
            # Generate data
            sample_list, label_list = self.generate_captcha_sample(20)
            images = []
            labels = []
            for s in sample_list:
                s = self.split_character_4(s)
                for ss in s:
                    images.append(np.asarray(ss))
            for l in label_list:
                for ll in l:
                    p = np.zeros(self.LABEL_NUM)
                    p[self.index_map[ll]] = 1.0
                    labels.append(p)

            self.tf_sess.run(self.tf_train_step, feed_dict={self.tf_image: images,
                                      self.tf_label: labels,
                                      self.tf_keep_prob: 1.0})

            if i % 100 == 0:
                train_accuracy = self.tf_sess.run(self.tf_accuracy, feed_dict={self.tf_image: images, self.tf_label: labels, self.tf_keep_prob: 0.8})
                print("step %d, training accuracy %g" % (i, train_accuracy))

            if i % 2000 == 0:
                self.tf_saver.save(self.tf_sess, "tmp/lastest.ckpt")

    def example_test(self):
        sample_list, label_list = self.generate_captcha_sample(1)
        images = []
        labels = []

        for s in sample_list:
            plt.imshow(np.asarray(s), cmap="gray")
            plt.show()
            splited = self.split_character_4(s)
            for ss in splited:
                images.append(np.asarray(ss))

        for l in label_list:
            for ll in l:
                p = np.zeros(self.LABEL_NUM)
                p[self.index_map[ll]] = 1.0
                labels.append(p)

        res = self.tf_sess.run(self.tf_predict,
                               feed_dict={self.tf_image: images, self.tf_label: labels, self.tf_keep_prob: 0.8})
        print(res)

    def restore_last_checkpoint(self):
        self.tf_saver.restore(self.tf_sess, "tmp/lastest.ckpt")

    def generate_character_sample(self, n):
        sample_list = []
        label_list = []
        for i in range(n):
            ch = random.choice(self.charset)
            img = self.captcha_generator.generate_image(ch)
            img = img.crop((0, 0, 32, 60))
            img = img.resize((self.DIGIT_IMAGE_WIDTH, self.DIGIT_IMAGE_HEIGHT))
            img = img.convert("L")
            sample_list.append(img)
            label_list.append(ch)
        return sample_list, label_list

    def generate_captcha_sample(self, n):
        sample_list = []
        label_list = []
        for i in range(n):
            s = ""
            for i in range(4):
                s = s + random.choice(self.charset)
            img = self.captcha_generator.generate_image(s)
            img = img.convert("L")
            sample_list.append(img)
            label_list.append(s)
        return sample_list, label_list

    def split_character_4(self, captcha:Image):
        if captcha.size != (160, 60):
            raise ValueError("Can only process 160*60 4 digit captcha image from module captcha.")

        n = 4
        width = 36
        padding = -6

        digitList = []
        for i in range(n):
            img = captcha.copy()
            img = img.crop(((width+padding)*i, 0, (width+padding)*i+width, 60))
            digitList.append(img)

        return digitList


if __name__ == "__main__":
    recognizer = CaptchaRecognizer()
    recognizer.restore_last_checkpoint()
    recognizer.train_with_generated_data()
    recognizer.example_test()

