"""

This is exactly the same model as in tensorflow MNIST Expert tutorial

Validation accuracy is around 0.88 (400 steps) to 0.90 (1800 steps)



Before you run the code:

1) install numpy, scipy, tensorflow

2) download emnist-letters.mat into data folder

"""



from os import path

from scipy.io import loadmat

import numpy as np

import tensorflow as tf



""" Helper Class """

class Data():

    @staticmethod

    def _one_hot(y):

        n = len(y)

        y_h = np.zeros(shape=(n, 26), dtype=np.float32)

        for i in range(n):

            y_h[i, y[i]-1] = 1.0

        return y_h



    def __init__(self, X, y):

        self.X = X

        self.y = self._one_hot(y)

        self.n = y.shape[0]

        self.cursor = self.n



    def next_batch(self, size):

        if self.cursor + size > self.n:

            self.cursor = 0

            ordering = np.random.permutation(self.n)

            self.X = self.X[ordering, :]

            self.y = self.y[ordering, :]

        x_b = self.X[self.cursor:(self.cursor + size), :]

        y_b = self.y[self.cursor:(self.cursor + size), :]

        self.cursor += size

        return x_b, y_b



""" Load and Split Data """

m = loadmat(path.join(path.dirname(__file__), "..", "data", "emnist-letters.mat"), struct_as_record=False)

x = m["dataset"][0, 0].train[0, 0].images

y = m["dataset"][0, 0].train[0, 0].labels.T[0]

x_t = x[1000:]

y_t = y[1000:]

x_v = x[:1000]

y_v = Data._one_hot(y[:1000])



""" Define Neural Network """

sy_x_b = tf.placeholder(tf.float32, shape=[None, 784])

sy_y_b = tf.placeholder(tf.float32, shape=[None, 26])

keep_prob = tf.placeholder(tf.float32)

images = tf.reshape(sy_x_b, shape=[-1, 28, 28, 1])



conv_1 = tf.layers.conv2d(images, filters=32, kernel_size=5, strides=(1, 1), padding="SAME", activation=tf.nn.relu, trainable=True)

pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")



conv_2 = tf.layers.conv2d(pool_1, filters=64, kernel_size=5, strides=(1, 1), padding="SAME", activation=tf.nn.relu, trainable=True)

pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")



flatten = tf.reshape(pool_2, shape=[-1, 64*7*7])

fc_1 = tf.layers.dense(flatten, units=1024, activation=tf.nn.relu, trainable=True)

fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

out = tf.layers.dense(fc_1_drop, units=26, activation=None, trainable=True)



y_hat = tf.nn.softmax(out)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=sy_y_b, logits=out)

step = tf.train.AdamOptimizer(5e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(sy_y_b, 1))

sy_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



""" Begin Training """

with tf.Session() as session:

    data = Data(x_t, y_t)

    session.run(tf.global_variables_initializer())

    print("training begin")

    for i in range(3000):

        x_b, y_b = data.next_batch(256)

        session.run(step, feed_dict={sy_y_b: y_b, sy_x_b: x_b, keep_prob: 0.5})

        if i % 10 == 0:

            accuracy = session.run(sy_accuracy,

                feed_dict={sy_y_b: y_v, sy_x_b: x_v, keep_prob: 1.0})

            print("step %d: validation accuracy %f" % (i, accuracy))