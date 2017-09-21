#!/usr/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
import lib
import pickle
from os.path import join


def load_sample(x):
    data = pickle.load(open(join("data/interim/bounding_box/", x), "rb"))
    return np.array(data['images']), np.array(data['labels'])


print("loading data ...")
test_data = load_sample("test")

print("building graph ...")
graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder(tf.float32, shape=(None, 50, 50))
    y = tf.placeholder(tf.float32, shape=(None, 4,))
    keep_prob = tf.placeholder(tf.float32)

    o = tf.reshape(x, (-1, 50, 50, 1))
    tf.summary.image("images", o, 1)
    o = lib.layers.convolutional(o, (5, 5, 1, 32), "first_layer")
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

feed_dict = {x: test_data[0], y: test_data[1], keep_prob: 1}
res = sess.run(o, feed_dict)

for xx in range(len(res)):
    ax.clear()
    ax.imshow(test_data[0][xx])

    """for i in range(len(res[xx])):
        res[xx][i] += 5 if i < 2 else 5"""

    print(res[xx])
    print(test_data[1][xx])
    rect = patches.Rectangle((res[xx][0], res[xx][1]), res[xx][2], res[xx][3], linewidth=1,
                             edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    x, y, w, h = test_data[1][xx][0], test_data[1][xx][1], test_data[1][xx][2], test_data[1][xx][3]

    rect = patches.Rectangle((x, y), w, h, linewidth=1,
                             edgecolor='w', facecolor='none')
    ax.add_patch(rect)
    plt.draw()
    plt.pause(0.1)
