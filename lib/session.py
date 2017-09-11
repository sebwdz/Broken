import tensorflow as tf


class Session:

    _name = None
    _graph = None

    _train_writer = None
    _test_writer = None

    _session = None

    def __init__(self, name, graph):
        self._name = name
        self._graph = graph

        self._train_writer = tf.summary.FileWriter(name + '/train', graph)
        self._test_writer = tf.summary.FileWriter(name + '/test')

        self._session = tf.InteractiveSession(graph=graph)

    def init(self):
        saver = tf.train.Saver()
        self._session.run(tf.global_variables_initializer())
        try:
            saver.restore(self._session, self._name + "/session")
        except:
            pass

    def train_step(self, train_step, merged, feed_dict, iteration):
        summary, _ = self._session.run([merged, train_step], feed_dict=feed_dict)
        self._train_writer.add_summary(summary, iteration)

    def test_step(self, accuracy, cross_entropy, merged, feed_dict, iteration):
        summary, train_accuracy, cost = self._session.run([merged, accuracy, cross_entropy], feed_dict=feed_dict)
        self._test_writer.add_summary(summary, iteration)
        return train_accuracy, cost

    def run(self, output, feed_dict):
        return self._session.run(output, feed_dict=feed_dict)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self._session, self._name + "/session")

