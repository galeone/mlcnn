#mlcnn: a multilayer convolutional neural network
#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""mitcbcl module wraps the MIT CBCL dataset into `dataset.DataSet` objects """

import glob
import math
import random
import numpy as np
from scipy.misc import imread
from dataset import DataSet


class MITCBCL(object):
    """ MITCBCL defines the method required to work with the MIT CBCL dataset """
    width = 19
    height = 19
    channel = 1
    num_class = 2

    def __init__(self, ds_dir):
        self._datasets = {"train": None, "test": None, "validation": None}

        for task in ["train", "test"]:
            images, labels = [], []
            for folder in ["face", "non-face"]:
                path = ds_dir + "/" + task + "/" + folder + "/*.pgm"
                label = 1 if folder == "face" else -1
                for image_path in glob.glob(path):
                    # scipy.misc.imread reads an image into a numpy array
                    images.append(imread(image_path))
                    labels.append(label)

            # keep some image from train dataset (and remove from it)
            # to build the validation dataset
            if task == "train":
                size = len(labels)
                random_indices = random.sample(
                    range(size), math.ceil(size / 3))
                validation_images = [images[i] for i in random_indices]
                validation_labels = [labels[i] for i in random_indices]

                for i in reversed(sorted(random_indices)):
                    del images[i]
                    del labels[i]

                self._datasets["validation"] = DataSet(validation_images,
                                                       validation_labels)
                print("Loaded {} validation image".format(self._datasets[
                    "validation"].length()))

            self._datasets[task] = DataSet(np.array(images), np.array(labels))
            print("Loaded {} {} image".format(self._datasets[task].length(),
                                              task))

    def get_train(self):
        """ get_train returns the train dataset """
        return self._datasets["train"]

    def get_test(self):
        """ get_test returns the test dataset """
        return self._datasets["test"]

    def get_validation(self):
        """ get_validation returns the validation dataset """
        return self._datasets["validation"]
