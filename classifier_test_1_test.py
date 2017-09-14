#!/usr/bin/python3.5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
import scipy.ndimage
import scipy.misc

import os
from os import listdir
from os.path import isfile, join


import lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_sample(directory, label, raw_dir):
    print(directory)
    files = [f for f in listdir(directory) if isfile(join(directory, f)) and ".json" not in f]
    images, labels, raw = [], [], []
    for file in files:
        images += [scipy.ndimage.imread(join(directory, file), flatten=True)]
        raw += [scipy.ndimage.imread(join(raw_dir, file), flatten=False)]
        images[-1] = scipy.misc.imresize(images[-1], (60, 60), interp='bilinear', mode=None)
        labels += [label]
    return images, labels, raw


def load_data():
    images, labels, raw = load_sample("data/interim/package_1/zoom_level_2/normal/",
                                      [1, 0], "data/raw/package_1/normal/")

    images2, labels2, raw2 = load_sample("data/interim/package_1/zoom_level_2/broken/",
                                         [0, 1], "data/raw/package_1/broken/")
    images += images2
    labels += labels2
    raw += raw2
    images, labels, raw = np.asarray(images), np.asarray(labels), np.asarray(raw)
    r = np.arange(images.shape[0])
    np.random.shuffle(r)
    return images[r], labels[r], raw[r]

print("loading data ...")
images, labels, raw = load_data()

all_images = images
all_labels = labels

images = np.array_split(np.asarray(images), 10)
labels = np.array_split(np.asarray(labels), 10)
raw = np.array_split(np.asarray(raw), 10)

graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder(tf.float32, shape=(None, 60, 60))
    y = tf.placeholder(tf.float32, shape=(None, 2,))
    keep_prob = tf.placeholder(tf.float32)

    o = tf.reshape(x, (-1, 60, 60, 1))
    tf.summary.image("images", o, 1)
    o = lib.layers.convolutional(o, (10, 10, 1, 32), "first_layer")
    o = lib.layers.pool(o, k_size=(1, 2, 2, 1), strides=(1, 2, 2, 1))
    o = lib.layers.convolutional(o, (5, 5, 32, 64), "second_layer")
    o = lib.layers.pool(o, k_size=(1, 4, 4, 1), strides=(1, 4, 4, 1))
    o = lib.layers.convolutional(o, (5, 5, 64, 128), "third_layer")
    o = lib.layers.pool(o, k_size=(1, 4, 4, 1), strides=(1, 4, 4, 1))
    o = tf.reshape(o, (-1, 2 * 2 * 128))
    o = lib.layers.fully_connected(o, (2 * 2 * 128, 200), "fully_connected")
    o = tf.nn.dropout(o, keep_prob)
    o = lib.layers.fully_connected(o, (200, 2), "output_layer", f=tf.nn.softmax)

plt.ion()
fig, ax = plt.subplots(1)

print("initializing session ...")
sess = lib.session.Session("models/classifier", graph=graph)
sess.init()

print("running...")
for ii in range(len(images)):
    batch = (images[ii], labels[ii], raw[ii])
    res = sess.run(o, feed_dict={x: batch[0], y: batch[1], keep_prob: 1})
    for i in range(len(res)):
        ax.clear()
        ax.imshow(batch[2][i])
        rect = patches.Rectangle((5, 5), batch[2][i].shape[1] - 10, batch[2][i].shape[0] - 10, linewidth=2,
                                 edgecolor='g' if res[i][0] > res[i][1] else 'r', facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((20, 20), batch[2][i].shape[1] - 40, batch[2][i].shape[0] - 40, linewidth=2,
                                 edgecolor='g' if batch[1][i][0] > batch[1][i][1] else 'r', facecolor='none')
        print("____")
        print(round(res[i][0], 4), round(res[i][1], 4))
        print(batch[1][i])
        ax.add_patch(rect)
        plt.draw()
        plt.pause(3)
