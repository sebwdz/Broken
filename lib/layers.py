import tensorflow as tf
import lib.summaries


def convolutional(t_input, dimension, layer_name, summaries=True, stdev=0.001, biais=0.01,
                  strides=(1, 1, 1, 1), padding='SAME', f=tf.nn.relu):

    with tf.name_scope(layer_name):

        w = tf.Variable(tf.truncated_normal(shape=dimension, stddev=stdev))
        b = tf.Variable(tf.constant(biais, shape=(dimension[3], )))

        if summaries:
            lib.summaries.variable_summaries(w)

        o = tf.nn.conv2d(t_input, w, strides=strides, padding=padding)
        return f(o + b)


def fully_connected(t_input, dimension, layer_name, summaries=True, stddev=0.001, biais=0.01, f=tf.nn.relu):

    with tf.name_scope(layer_name):

        w = tf.Variable(tf.truncated_normal(dimension, stddev=stddev))
        b = tf.Variable(tf.constant(biais, shape=(dimension[1], )))

        if summaries:
            lib.summaries.variable_summaries(w)

        return f(tf.matmul(t_input, w) + b)


def pool(t_input, f=tf.nn.max_pool, k_size=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME'):
    return f(t_input, ksize=k_size, strides=strides, padding=padding)
