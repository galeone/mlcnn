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
import tensorflow as tf
import mitcbcl
import utils

TRAINED_MODEL_FILENAME = "model.pb"


def test():
    """test the model"""

    # get the dataset
    dataset = mitcbcl.MITCBCL("mitcbcl")

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

        if __debug__:
            utils.print_graph_ops(graph)

        # extract tensor from graph
        unscaled_logits = graph.get_tensor_by_name("FC2/unscaled_logits:0")

        out_shape = tf.shape(unscaled_logits)

        # define some test operations
        labels = tf.placeholder(tf.float32, [None, 2], "labels")

        unscaled_logits = tf.reshape(unscaled_logits, [-1, 2])
        softmax = tf.nn.softmax(unscaled_logits, name="softmax")

        correct_predictions = tf.equal(
            tf.arg_max(softmax, 1), tf.arg_max(labels, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32),
            name="accuracy")

        with tf.Session() as sess:
            # test with the whole test dataset
            seq_size = 200
            test_images, test_labels = dataset.get_test().next_sequence(
                size=seq_size)
            accuracy_sum = 0.0
            batch_no = 1
            while test_labels != []:
                ul_value, softmax_value, accuracy_value = sess.run(
                    [unscaled_logits, softmax, accuracy],
                    feed_dict={
                        "input_batch:0": tf.reshape(
                            test_images, [-1, mitcbcl.MITCBCL.width,
                                          mitcbcl.MITCBCL.height,
                                          mitcbcl.MITCBCL.channel]).eval(),
                        "keep_prob:0": 1.0,
                        labels: test_labels
                    })

                print("Test for %d items (%d,%d). Accuracy: %g" %
                      (seq_size, seq_size * (batch_no - 1),
                       (seq_size *
                        (batch_no - 1)) + len(test_labels), accuracy_value))

                if __debug__:
                    print("unscaled_logits")
                    print(ul_value)
                    print("softmax")
                    print(softmax_value)

                accuracy_sum += accuracy_value
                batch_no += 1

                test_images, test_labels = dataset.get_test().next_sequence(
                    size=seq_size)

            print("Test accuracy: %g" % (accuracy_sum / batch_no))

        return 0


if __name__ == "__main__":
    sys.exit(test())
