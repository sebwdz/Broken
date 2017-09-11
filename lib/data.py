
from os import listdir
from os.path import isfile, join

import scipy.ndimage
import scipy.misc
import numpy as np


def load_force_labeled(directory, label, target, flatten=True):
    files = [f for f in listdir(directory) if isfile(join(directory, f)) and ".json" not in f]
    images, labels = [], []
    for file in files:
        images += [scipy.ndimage.imread(join(directory, file), flatten=flatten)]
        images[-1] = scipy.misc.imresize(images[-1], target, interp='bilinear', mode=None)
        labels += [label]
    return list(images), list(labels)


def load_many_labeled_directory(directories, function):
    inputs, labels = [], []
    for x in directories:
        tmp = function(x)
        inputs += tmp[0]
        labels += tmp[1]
    return np.asarray(inputs), np.asarray(labels)


def shuffle(inputs, labels):
    x = np.arange(inputs.shape[0])
    np.random.shuffle(x)
    return inputs[x], labels[x]


def batch(inputs, labels, batch_size):
    inputs = np.array_split(np.asarray(inputs), int(inputs.shape[0] / batch_size))
    labels = np.array_split(np.asarray(labels), int(labels.shape[0] / batch_size))
    return inputs, labels


def get_next_batch(images, labels):
    while True:
        for i in range(len(images)):
            yield images[i], labels[i]
