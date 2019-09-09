import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('D:\MNIST_data', one_hot=True)

n_hidden_1 = 256
n_hidden_2 = 128
n_hidden_3 = 128
n_input = 784
n_classes = 10

# INPUT AND OUTPUT
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# NETWORK PARAMETERS
stddev = 0.1
weights = {
    "w1": tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    "w2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    "w3": tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=stddev)),
    "out": tf.Variable(tf.random_normal([n_hidden_3, n_classes], stddev=stddev))
}
biases = {
    "b1": tf.Variable(tf.zeros([n_hidden_1], tf.float32)),
    "b2": tf.Variable(tf.zeros([n_hidden_2], tf.float32)),
    "b3": tf.Variable(tf.zeros([n_hidden_3], tf.float32)),
    "out": tf.Variable(tf.zeros([n_classes], tf.float32))
}


def network(inputs, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(inputs, weights['w1']), biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['w3']), biases['b3']))
    pre = tf.matmul(layer_3, weights['out']) + biases['out']
    return pre


pred = network(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))

init = tf.global_variables_initializer()

train_step = 500
batch_size = 100
display_step = 10

with tf.Session() as sess:
    sess.run(init)
    for k in range(train_step):
        loss = 0
        num_batch = int(mnist.train.num_examples / batch_size)
        for L in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
            _loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            loss += _loss
        if k % display_step == 0:
            _accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print('loss:%2f' % loss, '  accuracy:%2f' % _accuracy)
