#!/usr/bin/python3.5

import scipy.ndimage
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
import lib
import json

from PIL import Image


def load_sample():
    directory = "data/interim/package_2/zoom_level_1/normal/"

    file = directory + "bounding_box.json"

    labels_l = json.load(open(file))

    images, labels, raw_images = [], [], []

    for label in labels_l:
        filename = label["filename"]
        box = label["annotations"][0]
        img = Image.open(directory + filename)

        new_label = (box["x"], box["y"], box["width"], box["height"])
        img = np.asarray(img)
        raw_images.append(img)
        r1, r2 = img.shape[0] / 50.0, img.shape[1] / 50.0
        img = scipy.misc.imresize(img, (50, 50), interp='bilinear', mode=None)
        img = img[:, :, 1]
        new_label = (new_label[0] / r2, new_label[1] / r1, new_label[2] / r2, new_label[3] / r1)

        labels.append(new_label)
        images.append(img)
    return np.asarray(images), np.asarray(labels), raw_images


def load_data():
    images, labels, raw_images = load_sample()
    r = np.arange(images.shape[0])
    return images[r], labels[r], raw_images


print("loading data ...")
images, labels, raw_images = load_data()

all_images = images
all_labels = labels

images = [images]
labels = [labels]
raw_images = [raw_images]

print("building graph ...")
graph = tf.Graph()

with graph.as_default():

    x = tf.placeholder(tf.float32, shape=(None, 50, 50))
    y = tf.placeholder(tf.float32, shape=(None, 4,))
    keep_prob = tf.placeholder(tf.float32)

    o = tf.reshape(x, (-1, 50, 50, 1))
    tf.summary.image("images", o, 1)
    o = lib.layers.convolutional(o, (10, 10, 1, 32), "first_layer")
    o = lib.layers.pool(o, k_size=(1, 2, 2, 1), strides=(1, 2, 2, 1))
    o = lib.layers.convolutional(o, (5, 5, 32, 64), "second_layer")
    o = lib.layers.pool(o, k_size=(1, 2, 2, 1), strides=(1, 2, 2, 1))
    o = lib.layers.convolutional(o, (5, 5, 64, 128), "third_layer")
    o = lib.layers.pool(o, k_size=(1, 2, 2, 1), strides=(1, 2, 2, 1))
    o = tf.reshape(o, (-1, 7 * 7 * 128))
    o = lib.layers.fully_connected(o, (7 * 7 * 128, 200), "fully_connected")
    o = tf.nn.dropout(o, keep_prob)
    pos = lib.layer.fully_connected(o, (200, 2), "position_layer", f=tf.identity)
    size = lib.layers.fully_connected(o, (200, 2), "size_layer")
    o = tf.concat([pos, size], axis=1)

print("initializing session ...")
sess = lib.session.Session("models/bounding_box", graph=graph)
sess.init()
ii = 0

print("running...")

plt.ion()
fig, ax = plt.subplots(1)

for i in range(len(images)):
    batch = (images[ii], labels[ii], raw_images[ii])
    ii = ii + 1 if ii + 1 < len(images) else 0
    feed_dict = {x: batch[0], y: batch[1], keep_prob: 1}
    res = sess.run(o, feed_dict)

    for xx in range(len(res)):
        ax.clear()
        ax.imshow(batch[2][xx])

        r1, r2 = batch[2][xx].shape[0] / 50.0, batch[2][xx].shape[1] / 50.0
        x, y, w, h = res[xx][0] * r2, res[xx][1] * r1, res[xx][2] * r2, res[xx][3] * r1

        rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        x, y, w, h = batch[1][xx][0] * r2, batch[1][xx][1] * r1, batch[1][xx][2] * r2, batch[1][xx][3] * r1

        rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                 edgecolor='w', facecolor='none')
        ax.add_patch(rect)
        plt.draw()
        plt.pause(2)
