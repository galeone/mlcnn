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
import tensorflow as tf

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

        # unscaled_logits is a (?, 1, 1, 2) tensor, if the input image have the same size of the
        # image in the train set
        unscaled_logits = graph.get_tensor_by_name('FC2/unscaled_logits:0')

        out_shape = tf.shape(unscaled_logits)
        unscaled_logits = tf.reshape(unscaled_logits, [-1, 2])
        #back to original shape
        softmax = tf.reshape(tf.nn.softmax(unscaled_logits), out_shape)

        with tf.Session() as sess:
            start = datetime.now()
            spatial_output, unscaled = sess.run(
                [softmax, unscaled_logits],
                feed_dict={
                    "keep_prob:0": 1.0,
                    "input_batch:0": tf.reshape(image, [1, image.shape[
                        0], image.shape[1], 1]).eval()
                })
            end = datetime.now()
            print("spatial_output shape: {} from image with shape {}".format(
                spatial_output.shape, image.shape))
            print(spatial_output)
            print(unscaled)
            print(end - start)
        return 0


if __name__ == "__main__":
    sys.exit(test(sys.argv[1]))
