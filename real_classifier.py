#!/usr/bin/python3.5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import scipy.misc
import scipy.ndimage

import tensorflow as tf
import pickle
import os

import lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_dataset(x):
    data = []
    labels = []
    files = ['broken/R1922011', 'broken/dd3366',
             'broken/piege', 'broken/r_18',
             'broken/dd3409', 'broken/dd3533',
             'broken/dd3636', 'broken/R1092017',
             'normal/dd1382', 'normal/dd524',
             'normal/dd14', 'normal/dd69',
             'normal/dd112', 'normal/dd809',
             'normal/dd725', 'normal/dd719']
    for file in files:
        image = scipy.ndimage.imread(os.path.join("data/interim/generated/eq_faces/" + file) + ".jpg", flatten=True)
        data.append(scipy.misc.imresize(image, (100, 100), interp='bilinear', mode=None))
        labels.append((0, 1) if "broken" in file else (1, 0))
    return data, labels

print("loading data ...")
test_data = load_dataset("test")
print(len(test_data[0]))

graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder(tf.float32, shape=(None, 100, 100))
    y = tf.placeholder(tf.float32, shape=(None, 2,))
    keep_prob = tf.placeholder(tf.float32)

    o = tf.reshape(x, (-1, 100, 100, 1))
    tf.summary.image("images", o, 1)
    o = lib.layers.convolutional(o, (10, 10, 1, 32), "first_layer")
    o = lib.layers.pool(o, k_size=(1, 2, 2, 1), strides=(1, 2, 2, 1))
    o = lib.layers.convolutional(o, (5, 5, 32, 64), "second_layer")
    o = lib.layers.pool(o, k_size=(1, 4, 4, 1), strides=(1, 4, 4, 1))
    o = lib.layers.convolutional(o, (5, 5, 64, 128), "third_layer")
    o = lib.layers.pool(o, k_size=(1, 4, 4, 1), strides=(1, 4, 4, 1))
    o = tf.reshape(o, (-1, 4 * 4 * 128))
    o = lib.layers.fully_connected(o, (4 * 4 * 128, 200), "fully_connected")
    o = tf.nn.dropout(o, keep_prob)
    o = lib.layers.fully_connected(o, (200, 2), "output_layer", f=tf.nn.softmax)
    lib.summaries.variable_summaries(o)

plt.ion()
fig, ax = plt.subplots(1)

print("initializing session ...")
sess = lib.session.Session("models/classifier", graph=graph)
sess.init()

vp = 0
fp = 0

vn = 0
fn = 0

print("running...")
for ii in range(len(test_data)):
    batch = test_data
    res = sess.run(o, feed_dict={x: batch[0], y: batch[1], keep_prob: 1})
    for i in range(len(res)):
        ax.clear()
        ax.imshow(batch[0][i])
        rect = patches.Rectangle((5, 5), batch[0][i].shape[1] - 10, batch[0][i].shape[0] - 10, linewidth=2,
                                 edgecolor='g' if res[i][0] < res[i][1] else 'r', facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((20, 20), batch[0][i].shape[1] - 40, batch[0][i].shape[0] - 40, linewidth=4,
                                 edgecolor='g' if batch[1][i][0] < batch[1][i][1] else 'r', facecolor='none')

        print(res[i], batch[1][i])
# normal - normal (VN)
        if batch[1][i][0] > batch[1][i][1] and res[i][0] > res[i][1]:
            vn += 1
# normal - broken (FP)
        elif batch[1][i][0] > batch[1][i][1] and res[i][0] < res[i][1]:
            fp += 1
# broken - normal (FN)
        elif batch[1][i][0] < batch[1][i][1] and res[i][0] > res[i][1]:
            fn += 1
# broken - broken (VP)
        else:
            vp += 1

        ax.add_patch(rect)
        plt.draw()
        plt.pause(0.5)
    print("vp:", vp, " vn:", vn, "fp:", fp, "fn:", fn)
    exit(0)
