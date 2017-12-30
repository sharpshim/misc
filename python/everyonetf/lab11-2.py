import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
num_epochs = 2
batch_size = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# img : [NONE, 28*28]
X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
# L1
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding="SAME")
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

print L1
# Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# L2
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding="SAME")
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

print L2
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

L2 = tf.reshape(L2, [-1, 7*7*64])
# L3
W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2,W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(num_epochs) :
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_cost = 0
    for batch in range(total_batch) :
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {X : batch_x, Y : batch_y}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print "Epoch:", "%04d" % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost)

correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print ("Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.labels}))

import random
rand_idx = random.randint(0, mnist.test.num_examples-1)
print "Label : %d" % sess.run(tf.arg_max(mnist.test.labels[rand_idx:rand_idx+1], 1))
print "Prediction : %d" % sess.run(tf.arg_max(logits, 1), feed_dict={X : mnist.test.images[rand_idx:rand_idx+1]})
