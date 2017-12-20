import tensorflow as tf

from .. import config as cfg
from .capsnet_layer import CapsLayer


epsilon = 1e-9


class CapsNet(object):
    def __init__(self, input_size=[28, 28], output_size=10, is_training=True):
        self.graph = tf.Graph()
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.lambda_val = 0.5
        self.regularization_scale = 0.0005*784
        self.input_width = input_size[0]
        self.input_height = input_size[1]
        self.output_len = output_size

        with self.graph.as_default():
            if is_training:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, self.input_height, self.input_width, 1))
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
                self.Y = tf.one_hot(self.labels, depth=self.output_len, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                # t_vars = tf.trainable_variables()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            else:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, self.input_height, self.input_width, 1))
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
                self.Y = tf.zeros((cfg.batch_size, self.output_len))
                #self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, self.output_len, 1))
                self.build_arch()
                self.loss()
                self._summary()

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            assert conv1.get_shape() == [cfg.batch_size, self.input_height-8, self.input_width-8, 256]


        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
            # [batch_size, (w*h*8), 8, 1]

        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)
            # [batch_size, output_len, 16, 1]

        with tf.variable_scope('Masking'):
            # [batch_size, output_len, 1, 1]
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                                  axis=2, keep_dims=True) + epsilon)
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)

            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))

            self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, self.output_len, 1)))
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)

        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            assert fc1.get_shape() == [cfg.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [cfg.batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=(self.input_height*self.input_width), activation_fn=tf.sigmoid)

    def loss(self):
        # Margin loss

        max_l = tf.square(tf.maximum(0., self.m_plus - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - self.m_minus))
        # max_l: [cfg.batch_size, output_len, 1, 1]

        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))
        # max_l: [cfg.batch_size, output_len]

        T_c = self.Y
        L_c = T_c * max_l + self.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # Reconstruction loss
        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        self.total_loss = self.margin_loss + self.regularization_scale * self.reconstruction_err

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, self.input_height, self.input_height, 1))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
