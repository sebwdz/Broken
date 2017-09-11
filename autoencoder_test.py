
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import pickle


graph = tf.Graph()

L = 50


def load_sample(file):
    data = pickle.load(open(file, "rb"))
    return list(data["images"])


def load_data():
    images = load_sample("data/interim/bounding_box_little")
    images = images + load_sample("data/interim/bounding_box_big")
    images = np.asarray(images)
    r = np.arange(images.shape[0])
    np.random.shuffle(r)
    return images[r], images[r]


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

with graph.as_default():

    x = tf.placeholder(tf.float32, shape=(None, L, L))
    y = tf.placeholder(tf.float32, shape=(None, L, L))
    keep_prob = tf.placeholder(tf.float32)

    xx = tf.reshape(x, (-1, L, L, 1))

    with tf.name_scope("conv"):

        with tf.name_scope("layer1"):
            w = tf.Variable(tf.truncated_normal((L, L, 1, 50), stddev=0.001))
            b = tf.Variable(tf.constant(0.01))

            variable_summaries(w)

            o = tf.nn.conv2d(xx, w, strides=[1, 1, 1, 1], padding='SAME')
            o = tf.nn.relu(o + b)

        o = tf.nn.max_pool(o, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope("layer2"):
            w = tf.Variable(tf.truncated_normal((25, 25, 50, 100), stddev=0.001))
            b = tf.Variable(tf.constant(0.01))

            variable_summaries(w)

            o = tf.nn.conv2d(o, w, strides=[1, 1, 1, 1], padding='SAME')
            o = tf.nn.relu(o + b)

        features = tf.nn.max_pool(o, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope("deconv"):

        with tf.name_scope("layer1"):

            batch_size = tf.shape(features)[0]
            deconv_shape = tf.stack([batch_size, 25, 25, 50])

            w = tf.Variable(tf.truncated_normal((25, 25, 50, 100), stddev=0.001))
            b = tf.Variable(tf.constant(0.01))

            variable_summaries(w)

            o = tf.nn.conv2d_transpose(features, w, deconv_shape, [1, 2, 2, 1], padding='SAME')
            o = tf.nn.relu(o + b)

        with tf.name_scope("layer2"):

            batch_size = tf.shape(o)[0]
            deconv_shape = tf.stack([batch_size, 50, 50, 1])

            w = tf.Variable(tf.truncated_normal((50, 50, 1, 50), stddev=0.001))
            b = tf.Variable(tf.constant(0.01))

            variable_summaries(w)

            o = tf.nn.conv2d_transpose(o, w, deconv_shape, [1, 2, 2, 1], padding='SAME')
            o = tf.nn.relu(o + b)

    variable_summaries(o)

    o = tf.squeeze(o)

    o = tf.clip_by_value(o, 0.0, 255.0)

    variable_summaries(o)

    loss = tf.reduce_mean(tf.pow(y - o, 2))

    tf.summary.scalar('loss', loss)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    merged = tf.summary.merge_all()

images, labels = load_data()

all_images = images

images = np.array_split(np.asarray(images), 20)
labels = np.array_split(np.asarray(labels), 20)

test_images = images[0]
images = images[1:]

test_labels = labels[0]
labels = labels[1:]

plt.ion()
_, ax1 = plt.subplots(1)
_, ax2 = plt.subplots(1)

with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter("autoencoder" + '/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter("autoencoder" + '/test')
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, "autoencoder/session")
    except:
        pass

    ii = 0
    for i in range(1500):
        batch = (images[ii], labels[ii])
        ii = ii + 1 if ii + 1 < len(images) else 0
        if i % 10 == 0:
            summary, cost, res = sess.run([merged, loss, o], feed_dict={x: test_images, y: test_labels, keep_prob: 1})
            test_writer.add_summary(summary, i)
            print('step %d, cost %g' % (i, cost))

            for xx in range(1):
                ax1.imshow(res[xx])
                ax2.imshow(test_labels[xx])
                plt.draw()
                plt.pause(0.002)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
        train_writer.add_summary(summary, i)

    saver.save(sess, "autoencoder/session")