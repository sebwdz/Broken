#!/usr/bin/python3.5

import scipy.ndimage
import scipy.misc
import numpy as np
import begin
import tensorflow as tf
import lib
import json

import os
import os.path

from PIL import Image


def load_sample(source):
    directory = source

    files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f)) and ".json" not in f]
    images, labels, raw_images = [], [], []

    for filename in files:
        img = Image.open(os.path.join(directory, filename))

        raw_images.append((img, filename))
        img = np.asarray(img)
        img = scipy.misc.imresize(img, (50, 50), interp='bilinear', mode=None)
        img = img[:, :, 1]

        images.append(img)
    return np.asarray(images), raw_images


def load_data(source):
    images, raw_images = load_sample(source)
    r = np.arange(images.shape[0])
    return images[r], raw_images


@begin.start
def main(source, destination):

    if not os.path.exists(destination):
        os.makedirs(destination)

    print("loading data ...")
    images, raw_images = load_data(source)

    images = [images]
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
    sess = lib.session.Session("bounding_box", graph=graph)
    sess.init()
    ii = 0

    print("running...")

    for i in range(len(images)):
        batch = (images[ii], raw_images[ii])
        ii = ii + 1 if ii + 1 < len(images) else 0
        feed_dict = {x: batch[0], keep_prob: 1}
        res = sess.run(o, feed_dict)

        for xx in range(len(res)):

            img = np.asarray(batch[1][xx][0])

            r1, r2 = img.shape[0] / 50.0, img.shape[1] / 50.0
            x, y, w, h = res[xx][0] * r2, res[xx][1] * r1, res[xx][2] * r2, res[xx][3] * r1

            crop_img = batch[1][xx][0].crop((int(x), int(y), int(w) + int(x), int(h) + int(y)))
            crop_img.save(os.path.join(destination, batch[1][xx][1]))

