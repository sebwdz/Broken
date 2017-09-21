#!/usr/bin/python3.5

import tensorflow as tf
import pickle

import lib

import os.path
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_dataset(x):
    data = pickle.load(open(os.path.join("data/interim/bounding_box/", x), "rb"))
    return np.array(data['images']), np.array(data['labels'])

print("loading data ...")
next_batch = lib.data.get_next_batch(*lib.data.batch(*lib.data.shuffle(*load_dataset("train")), 10))
test_data = load_dataset("test")

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
    lib.summaries.variable_summaries(pos)
    lib.summaries.variable_summaries(size)

    with tf.name_scope("cross_entropy"):
        loss = tf.reduce_sum(tf.pow(o - y, 2)) / (2.0 * tf.cast(tf.shape(x)[0], tf.float32))
        cross_entropy = loss
        tf.summary.scalar('IoU_loss', loss)

    with tf.name_scope("accuracy"):
        xA, yA = tf.maximum(y[:, 0], pos[:, 0]), tf.maximum(y[:, 1], pos[:, 1])
        xB, yB = (tf.minimum(y[:, 0] + y[:, 2], pos[:, 0] + size[:, 0]),
                  tf.minimum(y[:, 0] + y[:, 3], pos[:, 1] + size[:, 1]))

        interArea = (xB - xA) * (yB - yA)

        boxAArea = tf.maximum(y[:, 2], 0.0) * tf.maximum(y[:, 3], 0.0)
        boxBArea = tf.maximum(size[:, 0], 0.0) * tf.maximum(size[:, 1], 0.0)

        d = tf.maximum(0.0, boxAArea + boxBArea - interArea)

        accuracy_f = tf.where(
            tf.equal(d, 0.0),
            tf.zeros_like(interArea), tf.truediv(interArea, d))

        accuracy = tf.reduce_mean(accuracy_f)
        tf.summary.scalar('accuracy', accuracy)

    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

    merged = tf.summary.merge_all()

print("initializing session ...")
sess = lib.session.Session("models/bounding_box", graph=graph)
sess.init()
ii = 0

print("running...")
for i in range(1000):
    batch = next_batch.__next__()
    if i % 50 == 0:
        feed_dict = {x: test_data[0], y: test_data[1], keep_prob: 1}
        train_accuracy, cost = sess.test_step(accuracy, cross_entropy, merged, feed_dict, i)
        print('step %d, test accuracy %g, cost %g' % (i, train_accuracy, cost))
    feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.75}
    sess.train_step(train_step, merged, feed_dict, i)

print("saving ...")
sess.save()
