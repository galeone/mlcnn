#mlcnn: a multilayer convolutional neural network
#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""run train and tests for mlcnn model"""

import sys
from datetime import datetime
from scipy.misc import imread
import numpy as np
import tensorflow as tf
import dataset

TRAINED_MODEL_FILENAME = "model.pb"


def test(path):
    """test the model"""

    # get the grayscale image
    image = imread(path, flatten=True)

    # import the model into the current graph
    with tf.Graph().as_default() as graph:

        const_graph_def = tf.GraphDef()
        with open(TRAINED_MODEL_FILENAME, 'rb') as saved_graph:
            const_graph_def.ParseFromString(saved_graph.read())
            # replace current graph with the saved graph def (and content)
            # name="" is importat because otherwise (with name=None)
            # the graph definitions will be prefixed with import.
            # eg: the defined operation FC2/unscaled_logits:0
            # will be import/FC2/unscaled_logits:0
            tf.import_graph_def(const_graph_def, name="")

        # get the output of the L2 layer (maxpool)
        # the input image has been resized from 19x19 to 19/4 x 19/4
        # using 2 maxpooling 2x2 with stride 2. Rounded to 5x5
        # mp2 is a tensor with size (?, 5, 5, 64)
        mp2 = graph.get_tensor_by_name("L2/L2:0")

        # W_fc1 is a (5x5x64, 1024) tensor
        W_fc1_orig = graph.get_tensor_by_name('FC1/W_fc1:0')

        W_fc1 = tf.reshape(W_fc1_orig, [5, 5, 64, 1024])

        # fc_to_conv is a 1,1,1024 tensor results of a convoltion (padding=VALID)
        # a convolution of 5x5x64 tensor with a 5x5x64, without padding, resultis in a 1x1x1024 tensor

        # to achieve the same effect of the FC, remeber to add the ReLU layer!
        h_fc1_to_conv = tf.nn.relu(tf.nn.conv2d(
            mp2, W_fc1, [1, 1, 1, 1], padding='VALID') +
                                   graph.get_tensor_by_name('FC1/b_fc1:0'))

        # get  W_fc2, reshape it from 1024,2 to 1,1,1024,2
        W_fc2 = graph.get_tensor_by_name('FC2/W_fc2:0')

        W_fc2 = tf.reshape(W_fc2, [1, 1, 1024, 2])

        # unscaled_logits_conv is a (?, 1, 1, 2) tensor, if the input image have the same size of the
        # image in the train set
        unscaled_logits_conv = tf.nn.conv2d(
            h_fc1_to_conv, W_fc2, [1, 1, 1, 1],
            padding='VALID') + graph.get_tensor_by_name('FC2/b_fc2:0')

        unscaled_logits_conv = tf.reshape(unscaled_logits_conv, [-1, 2])
        softmax_conv = tf.nn.softmax(unscaled_logits_conv)

        with tf.Session() as sess:
            start = datetime.now()
            heatmap = sess.run(
                softmax_conv,
                feed_dict={
                    "keep_prob:0": 1.0,
                    "input_batch:0": tf.reshape(image, [1, image.shape[
                        0], image.shape[1], 1]).eval()
                })
            print("heatmap shape: {} from image with shape {}".format(
                heatmap.shape, image.shape))

            """
            # convert one hot vector representation to a numeric value
            dummy_dataset = dataset.DataSet(
                np.array([0, 0]), np.array([1, -1]))

            real_heatmap = []
            for i in range(0, heatmap.shape[0]):
                real_heatmap.append(dummy_dataset.one_hot_vector_to_label(
                    heatmap[i]))

            print(real_heatmap)
            print(len(real_heatmap))
            """
            print(heatmap)
            end = datetime.now()
            print(end - start)
        return 0


if __name__ == "__main__":
    sys.exit(test(sys.argv[1]))
