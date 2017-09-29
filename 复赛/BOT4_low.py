import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim


class Test():

    def __init__(self, l2_reg_lambda=0.001):
        print('*' * 5 + 'version 1' + '*' * 5)
        self.l2_reg_lambda = l2_reg_lambda
        self.input_x = tf.placeholder(tf.float32, [None, 300, 100, 3], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, 3], name='input_y')
        """3x7(days) + 2x1(指标,price) = 23"""
        self.others = tf.placeholder(tf.float32, [None, 23], name='input_other')
        self.keep_prob = tf.placeholder('float', name='dropout_keep_prob')
        # slice
        input_top = tf.slice(self.input_x, [0, 0, 0, 0], [-1, 160, -1, -1], name='top_input')
        input_center = tf.slice(self.input_x, [0, 190, 0, 0], [-1, 50, -1, -1], name='center_input')
        input_bottom = tf.slice(self.input_x, [0, 245, 0, 0], [-1, 55, -1, -1], name='bottom_input')
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(self.l2_reg_lambda)):
            # top_input
            with tf.name_scope("top"):
                # 1x1 layer -> 160 x 11 layer: 160x100-> 1x15*128
                branch1x1 = slim.conv2d(input_top, 128, [1, 1],
                                        padding='SAME', scope="top-1x1")
                branch10x100 = slim.conv2d(branch1x1, 76, [160, 11],
                                           stride=6, padding='VALID',
                                           scope="top-160x11")
                # pool layer 1x15x64
                branch_pool = slim.pool(input_top, kernel_size=[160, 11], stride=6,
                                        pooling_type="AVG", scope='top-pool')
                branch_pool = slim.conv2d(branch_pool, 64, [1, 1], padding='SAME',
                                          scope="top-pool-1x1")
                net1 = tf.concat(axis=3, values=[branch10x100, branch_pool])
            # center_input
            with tf.name_scope("center"):
                # 1x1 layer -> 50 x 11 layer: 50x100-> 1x15x64
                branch1x1 = slim.conv2d(input_center, 64, [1, 1],
                                        padding='SAME', scope="center-1x1")
                branch10x100 = slim.conv2d(branch1x1, 32, [50, 11],
                                           stride=6, padding='VALID',
                                           scope="center-50x11")
                # pool layer -> 1x15x32
                branch_pool = slim.pool(input_center, kernel_size=[50, 11], stride=6,
                                        pooling_type="AVG", scope='center-pool')
                branch_pool = slim.conv2d(branch_pool, 32, [1, 1], padding='SAME',
                                          scope="center-pool-1x1")
                net2 = tf.concat(axis=3, values=[branch10x100, branch_pool])
            # bottom_input
            with tf.name_scope("bottom"):
                # 1x1 layer -> 55 x 11 layer: 55x100-> 1x15x64
                branch1x1 = slim.conv2d(input_bottom, 32, [1, 1],
                                        padding='SAME', scope="bottom-1x1")
                branch10x100 = slim.conv2d(branch1x1, 28, [55, 11],
                                           stride=6, padding='VALID',
                                           scope="bottom-55x11")
                # pool layer -> 1x15x32
                branch_pool = slim.pool(input_bottom, kernel_size=[55, 11], stride=6,
                                        pooling_type="AVG", scope='bottom-pool')
                branch_pool = slim.conv2d(branch_pool, 28, [1, 1], padding='SAME',
                                          scope="bottom-pool-1x1")
                net3 = tf.concat(axis=3, values=[branch10x100, branch_pool])
            net = tf.concat(axis=3, values=[net1, net2, net3])
            net = slim.flatten(net, scope="flatten")
            net = slim.fully_connected(net, 1024, scope='fc_1')
            net = tf.nn.dropout(net, self.keep_prob)
            net = tf.concat(axis=1, values=[net, self.others])
            net = slim.fully_connected(net, 3, scope='fc_2',
                                       activation_fn=tf.nn.tanh)
        with tf.name_scope("predictions"):
            self.predictions = tf.multiply(0.15, net, name='predictions')
        with tf.name_scope("loss"):
            losses = tf.sqrt(tf.square(self.predictions - self.input_y) + 1e-15) - 0.2 * tf.sign(self.predictions * self.input_y)
            self.loss = tf.reduce_mean(losses + tf.reduce_sum(slim.losses.get_regularization_losses()))
        with tf.name_scope("score"):
            self.score = tf.reduce_mean(tf.sqrt(tf.square(self.predictions - self.input_y)))
