#!/usr/bin/python3.5

import tensorflow as tf
import os

import pickle
import numpy as np
import lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_dataset(x):
    data = pickle.load(open(os.path.join("data/interim/classifier/", x), "rb"))
    return np.array(data['images']), np.array(data['labels'])

print("loading data ...")
next_batch = lib.data.get_next_batch(*lib.data.batch(*lib.data.shuffle(*load_dataset("train")), 10))
test_data = load_dataset("test")

print("building graph ...")
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

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=o))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_sum(tf.pow(o - y, 2)) / (2.0 * tf.cast(tf.shape(x)[0], tf.float32))
        """correct_prediction = tf.equal(tf.argmax(o, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))"""
        tf.summary.scalar('accuracy', accuracy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    merged = tf.summary.merge_all()

print("initializing session ...")
sess = lib.session.Session("models/regresor", grxaph=graph)
sess.init()
ii = 0

print("running...")
for i in range(1000):
    batch = next_batch.__next__()
    if i % 10 == 0:
        feed_dict = {x: test_data[0], y: test_data[1], keep_prob: 1}
        train_accuracy, cost = sess.test_step(accuracy, cross_entropy, merged, feed_dict, i)
        print('step %d, test accuracy %g, cost %g' % (i, train_accuracy, cost))
    feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.75}
    sess.train_step(train_step, merged, feed_dict, i)

print("saving ...")
sess.save()
