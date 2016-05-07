#mlcnn: a multilayer convolutional neural network
#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""dataset module contains utilities to work with datasets"""

import random
import numpy as np


class DataSet(object):
    """
    DataSet represents a DataSet of images and labels.
    Labels are always converted to a one hot vector.
    """

    def __init__(self, images, labels):
        self._len = len(images)
        assert self._len == len(labels)

        self._images = np.array(images)
        self._labels = np.array(labels)

        # LUT:
        # access with the dataset label
        # exit with the one-hot-vector associated
        self._one_hot_vector = {}
        unique_labels = np.unique(self._labels)
        # encode each label with a specific position in a vector
        # of self._len elements
        for idx, label in enumerate(unique_labels):
            self._one_hot_vector[str(label)] = [
                1 if i == idx else 0 for i in range(unique_labels.size)
            ]

        self._current_position = -1

    def get_data(self):
        """returns the whole dataset
        @return: [images], [labels]
        """
        images = np.array([self._images[i] for i in range(self._len)])
        labels = np.array([self._one_hot_vector[str(self._labels[i])]
                           for i in range(self._len)])

        return images, labels

    def next_sequence(self, size=100):
        """returns the next amount of [image],[label] if present"""
        items = self._current_position + size if self._current_position + size < self._len else self._len - self._current_position

        images = np.array([self._images[i]
                           for i in range(self._current_position, items)])

        labels = np.array([self._one_hot_vector[str(self._labels[i])]
                           for i in range(self._current_position, items)])

        self._current_position += size

        return images, labels

    def next_batch(self, batch_size=10):
        """
        next_batch returns a random batch of element from the dataset
        @return: [images], [labels]
        """

        images, labels = np.array([]), np.array([])
        random_indices = random.sample(range(self._len), batch_size)

        images = np.array([self._images[i] for i in random_indices])
        labels = np.array([self._one_hot_vector[str(self._labels[i])]
                           for i in random_indices])

        return images, labels

    def length(self):
        """ return the itemset cardinality """
        return self._len

    def one_hot_vector_to_label(self, ohv):
        """ return the original label associated to the one hot vector representation """
        for oldlabel, onehot in self._one_hot_vector.items():
            if np.all(ohv == onehot):
                return int(oldlabel)
        return False
