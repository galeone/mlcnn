#mlcnn: a multilayer convolutional neural network
#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""run train and tests for mlcnn model"""

import os
import sys
import tensorflow as tf
import freeze_graph
import model
import mitcbcl

SESSION_DIR = "session"
SUMMARY_DIR = "summary"
TRAINED_MODEL_FILENAME = "model.pb"


def loss(unscaled_logits_reshaped, labels):
    """loss defines the loss functions used in the model
    labels is the tensor (one-hot-vector) that contains the real classes
    unscaled_logits_respahed is the last layer output, reshaped to (?, 2)
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        unscaled_logits_reshaped,
        labels, name='cross_entropy')
    loss_fn = tf.reduce_mean(cross_entropy, name="mean_cross_entropy")
    return loss_fn


def train_step(learning_rate, loss_fn, global_step):
    """ train_step returns the train step.
    global_step must be a tensorflow variable
    """
    return tf.train.AdamOptimizer(learning_rate).minimize(
        loss_fn, global_step=global_step)


def train():
    """train model"""

    # if the trained model does not exist
    if not os.path.exists(TRAINED_MODEL_FILENAME):
        dataset = mitcbcl.MITCBCL("mitcbcl")

        # train graph is the graph that contains the variable
        train_graph = tf.Graph()

        # create a scope for the train_graph
        with train_graph.as_default():
            ##### tensor flow graph input (placeholders)
            images = tf.placeholder(
                tf.float32,
                # undefined amount of images in batch
                # images with width x height x depth
                shape=[None, mitcbcl.MITCBCL.width, mitcbcl.MITCBCL.height,
                       mitcbcl.MITCBCL.channel],
                name='input_batch')

            # one hot vector
            labels = tf.placeholder(tf.float32,
                                    shape=[None, dataset.num_class],
                                    name="labels")

            keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            ##### build the model

            # infer the probability of images to appartain to certain classes
            unscaled_logits, model_summaries = model.get(images, keep_prob)
            unscaled_logits_reshaped = tf.reshape(unscaled_logits, [-1, 2])

            ##### train operations and parameters
            batch_size = 50
            min_accuracy = 0.9
            max_iterations = 10**100 + 1
            display_step = 100

            learning_rate = 1e-5
            loss_op = loss(unscaled_logits_reshaped, labels)
            global_step = tf.Variable(0, trainable=False)
            train_op = train_step(learning_rate, loss_op, global_step)

            #### metrics to evaluate the model
            # predictions are the probability values, that softmax returns
            predictions = tf.nn.softmax(unscaled_logits_reshaped,
                                        name="softmax")
            correct_predictions = tf.equal(
                tf.arg_max(predictions, 1), tf.arg_max(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            #### define summaries
            # accuracy is shared between train and test summaries
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)
            train_summary_op = tf.merge_summary([
                accuracy_summary,
                tf.scalar_summary("loss", loss_op),
                #unpack model_summaries
                *model_summaries
            ])

            # tensor flop operator to initialize all the variables in a session
            init_op = tf.initialize_all_variables()

            with tf.Session() as sess:
                sess.run(init_op)

                # create a saver: to store current computation and restore the graph
                # useful when the train step has been interrupeted
                saver = tf.train.Saver()

                # restore previous session.
                # if the model is trained (the accuracy has already been reached)
                # the train step is skieed. Otherwise train the model and test

                checkpoint = tf.train.get_checkpoint_state(SESSION_DIR)
                if checkpoint and checkpoint.model_checkpoint_path:
                    saver.restore(sess, checkpoint.model_checkpoint_path)
                else:
                    print("[!] Unable to restore from checkpoint",
                          file=sys.stderr)

                # validation test set
                validation_images, validation_labels = dataset.get_validation(
                ).get_data()

                validation_images_reshaped = tf.reshape(
                    validation_images, [-1, mitcbcl.MITCBCL.width,
                                        mitcbcl.MITCBCL.height,
                                        mitcbcl.MITCBCL.channel]).eval()

                train_summary_writer = tf.train.SummaryWriter(
                    SUMMARY_DIR + "/train",
                    graph=sess.graph)

                for step in range(max_iterations):
                    batch_images, batch_labels = dataset.get_train(
                    ).next_batch(batch_size)

                    batch_images_reshaped = tf.reshape(
                        batch_images, [-1, mitcbcl.MITCBCL.width,
                                       mitcbcl.MITCBCL.height,
                                       mitcbcl.MITCBCL.channel]).eval()

                    #train
                    _, loss_val = sess.run(
                        [train_op, loss_op],
                        feed_dict={
                            images: batch_images_reshaped,
                            labels: batch_labels,
                            keep_prob: 0.5
                        })

                    if step % display_step == 0 and step > 0:
                        gs_value, accuracy_value, summary_line = sess.run(
                            [global_step, accuracy, train_summary_op],
                            feed_dict={images: validation_images_reshaped,
                                       labels: validation_labels,
                                       keep_prob: 1.0})
                        print(
                            "Global step: %d. Validation accuracy: %g. Loss: %g"
                            % (gs_value, accuracy_value, loss_val))

                        # create summary for this train step
                        # global_step in add_summary is the local step (thank you tensorflow)
                        train_summary_writer.add_summary(summary_line,
                                                         global_step=step)

                        # save the current session (until this step) in the session dir
                        # export a checkpint in the format SESSION_DIR/model-<global_step>.meta
                        # always use 0 in as global step in order to
                        # have only one file in the folder
                        saver.save(sess, SESSION_DIR + "/model", global_step=0)

                        if accuracy_value > min_accuracy:
                            break

                # end of train

                # save train summaries to disk
                train_summary_writer.flush()

                # save model skeleton (the empty graf, its definition)
                tf.train.write_graph(train_graph.as_graph_def(),
                                     SESSION_DIR,
                                     "skeleton.pb",
                                     as_text=False)

                freeze_graph.freeze_graph(SESSION_DIR + "/skeleton.pb", "",
                                          True, SESSION_DIR + "/model-0",
                                          "FC2/unscaled_logits",
                                          "save/restore_all", "save/Const:0",
                                          TRAINED_MODEL_FILENAME, False, "")
    else:
        print("Trained model %s already exits" % (TRAINED_MODEL_FILENAME))
    return 0


if __name__ == "__main__":
    sys.exit(train())
