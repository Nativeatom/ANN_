""" Experiments for Regular and Adaptive Network



Before you run the code:

1) install numpy, scipy, tensorflow

2) download emnist-letters.mat into data folder

3) process emnist-letters.mat using process.py

"""

from os import path

from scipy.io import loadmat

import numpy as np

import tensorflow as tf

m = loadmat(path.join(path.dirname(__file__), "..", "data", "emnist-mini.mat"))

x_t = m["x_t"]

y_t = m["y_t"]

x_v = m["x_v"]

y_v = m["y_v"]

INPUT_DIM = 784

HIDDEN_UNITS = 256

OUTPUT_DIM = 26

T_SIZE = y_t.shape[0]

V_SIZE = y_v.shape[0]


def tf_dense(inputs, units, name, activation=tf.nn.relu, trainable=True):
    return tf.layers.dense(

        inputs=inputs, units=units, activation=activation, use_bias=True, name=name,

        kernel_initializer=tf.contrib.layers.xavier_initializer(),

        bias_initializer=tf.truncated_normal_initializer(), trainable=trainable)


def regular():
    sy_x_b = tf.placeholder(tf.float32, shape=[None, INPUT_DIM], name="x_b")

    sy_y_b = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM], name="y_b")

    W_1 = tf.get_variable("W_1", shape=[INPUT_DIM, HIDDEN_UNITS], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())

    b_1 = tf.get_variable("b_1", shape=[HIDDEN_UNITS], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

    sy_h = tf.nn.relu(tf.matmul(sy_x_b, W_1) + b_1)

    W_2 = tf.get_variable("W_2", shape=[HIDDEN_UNITS, OUTPUT_DIM], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())

    b_2 = tf.get_variable("b_2", shape=[OUTPUT_DIM], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

    logits = tf.matmul(sy_h, W_2) + b_2

    sy_y_p = tf.nn.softmax(logits, name="y_p")

    accuracy = tf.reduce_mean(tf.cast(tf.equal(

        tf.argmax(sy_y_b, 1), tf.argmax(sy_y_p, 1)), tf.float32), name="accuracy")

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=sy_y_b, logits=logits)

    step = tf.train.AdamOptimizer(1e-3).minimize(loss, name="step")

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(2000):

            sess.run(step, feed_dict={sy_y_b: y_t, sy_x_b: x_t})

            if i % 10 == 0:
                a = sess.run(accuracy, feed_dict={sy_y_b: y_v, sy_x_b: x_v})

                print("step %d: validation accuracy %f" % (i, a))


def ann(LAMBDA=5e-1):
    sy_x_b = tf.placeholder(tf.float32, shape=[None, INPUT_DIM], name="x_b")

    sy_y_b = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM], name="y_b")

    W_1 = tf.get_variable("W_1", shape=[INPUT_DIM, HIDDEN_UNITS], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())

    b_1 = tf.get_variable("b_1", shape=[HIDDEN_UNITS], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

    fc_1 = tf.matmul(sy_x_b, W_1) + b_1

    sy_h_t = tf.get_variable("h_t", shape=[T_SIZE, HIDDEN_UNITS], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer())

    sy_h_c = tf.get_variable("h_c", shape=[V_SIZE, HIDDEN_UNITS], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer())

    W_2 = tf.get_variable("W_2", shape=[HIDDEN_UNITS, OUTPUT_DIM], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())

    b_2 = tf.get_variable("b_2", shape=[OUTPUT_DIM], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

    logits_t = tf.matmul(tf.nn.relu(sy_h_t), W_2) + b_2

    logits_c = tf.matmul(tf.nn.relu(sy_h_c), W_2) + b_2

    sy_y = tf.get_variable("y", shape=[V_SIZE, OUTPUT_DIM], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer())

    loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sy_y_b, logits=logits_t)) + \
             LAMBDA * tf.reduce_mean(tf.square(tf.nn.relu(sy_h_t) - fc_1))

    loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(sy_y), logits=logits_c)) + \
             LAMBDA * tf.reduce_mean(tf.square(tf.nn.relu(sy_h_c) - fc_1))

    step_t = tf.train.AdamOptimizer(1e-3).minimize(loss_t, name="step_t", var_list=[W_1, b_1, W_2, b_2, sy_h_t])

    step_c = tf.train.AdamOptimizer(1e-3).minimize(loss_c, var_list=[sy_h_c, sy_y], name="step_c")

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(sy_y_b, 1), tf.argmax(tf.nn.softmax(sy_y), 1)), tf.float32), name="accuracy")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for j in range(5):

            for i in range(501):

                sess.run(step_t, feed_dict={sy_y_b: y_t, sy_x_b: x_t})

                if i % 100 == 0:
                    tl = sess.run(loss_t,

                                  feed_dict={sy_y_b: y_t, sy_x_b: x_t})

                    print("step %.03d: training loss %f" % (i, tl))

            for i in range(501):

                sess.run(step_c, feed_dict={sy_y_b: y_v, sy_x_b: x_v})

                if i % 100 == 0:
                    tl = sess.run(loss_c,

                                  feed_dict={sy_y_b: y_v, sy_x_b: x_v})

                    print("step %.03d: classify loss %f" % (i, tl))

            a = sess.run(accuracy, feed_dict={sy_y_b: y_v, sy_x_b: x_v})

            print("  validation accuracy %f" % a)

regular()