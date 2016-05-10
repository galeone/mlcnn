#mlcnn: a multilayer convolutional neural network
#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""model contains the mlcnn model and utilities functions"""

import tensorflow as tf
import utils


def get(x, keep_prob):
    """
    input:
        x: placeholder for the image batch [None, widht, height, depth]
        keep_prob: dropout probability
    returns:
        unbounded logits
        [summaries]
    """

    summaries = []

    # define weights and biases for first conv layer
    with tf.variable_scope('L1'):
        # input: the placeholder x [None, width ,height]
        # output: 32 features for each 5x5 window
        W1, W1_s = utils.kernels([5, 5, 1, 32], name="W1")
        b1, b1_s = utils.bias([32], name="b1")
        summaries += W1_s
        summaries += b1_s

        h1 = tf.nn.relu(
            tf.add(
                tf.nn.conv2d(input=x,
                             filter=W1,
                             strides=[1, 1, 1, 1],
                             padding='SAME'),
                b1),
            name="ReLU")

        mp1 = tf.nn.max_pool(value=h1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME',
                             name='out')
        # now the "image" is (width/2)x(height/2)

    with tf.variable_scope('L2'):
        W2, W2_s = utils.kernels([5, 5, 32, 64], name="W2")
        b2, b2_s = utils.bias([64], name="b2")
        summaries += W2_s
        summaries += b2_s

        h2 = tf.nn.relu(
            tf.add(
                tf.nn.conv2d(input=mp1,
                             filter=W2,
                             strides=[1, 1, 1, 1],
                             padding='SAME'),
                b2),
            name="ReLU")
        mp2 = tf.nn.max_pool(value=h2,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME',
                             name='out')
        # now the "image" is (width/4)x(height/4) (x 64)

    with tf.variable_scope('FC1'):
        # flatter every image in a row of length: (width/4)*(height/4)*64
        # -1 means: automatically find the number of rows (that's the batch size btw)
        W_fc1, W_fc1_s = utils.weight([5, 5, 64, 1024], name="W_fc1")
        b_fc1, b_fc1_s = utils.bias([1024], name="b_fc1")

        summaries += W_fc1_s
        summaries += b_fc1_s

        h_fc1 = tf.nn.relu(
            tf.add(
                tf.nn.conv2d(mp2, W_fc1, [1, 1, 1, 1],
                             padding='VALID'),
                b_fc1),
            name='h_fc1')

        # add dropout in training in order to reduce overfitting
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='out')

    with tf.variable_scope('FC2'):
        # readout layer
        W_fc2, W_fc2_s = utils.weight([1, 1, 1024, 2], name="W_fc2")
        b_fc2, b_fc2_s = utils.bias([2], name="b_fc2")

        summaries += W_fc2_s
        summaries += b_fc2_s

        unscaled_logits = tf.add(
            tf.nn.conv2d(h_fc1_drop,
                         W_fc2, [1, 1, 1, 1],
                         padding='VALID'),
            b_fc2,
            name="unscaled_logits")
        #uncaled logits is a 1x1x1x2 tensor if the input image is a 19x19 image
        return unscaled_logits, summaries
