import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
num_epochs = 30
batch_size = 100

# read mnist data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# init X, Y placeholder
# img : [NONE, 28*28]
X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(dtype=tf.float32)

# L1 Layer conv2d --> relu --> maxpool
W1 = tf.Variable(tf.random_normal([3, 3, 1, 8], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding="SAME")
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
L1 = tf.nn.dropout(L1, keep_prob)

#print L1
# Tensor("MaxPool:0", shape=(?, 14, 14, 8), dtype=float32)

# L2 Layer conv2d --> relu --> maxpool
W2 = tf.Variable(tf.random_normal([3,3,8,16], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding="SAME")
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
L2 = tf.nn.dropout(L2, keep_prob)
#print L2
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 16), dtype=float32)

# L3 Layer conv2d --> relu --> maxpool
W3 = tf.Variable(tf.random_normal([3,3,16,32], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding="SAME")
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
L3 = tf.nn.dropout(L3, keep_prob)
# print L3
# Tensor("dropout_2/mul:0", shape=(?, 4, 4, 32), dtype=float32)

# L3 Layer fully connected
L3 = tf.reshape(L3, [-1, 4*4*32])
W4 = tf.get_variable("W4", shape=[4*4*32, 10],  initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal(shape=[10]))
logits = tf.matmul(L3, W4) + b4

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# make session and initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for each epoch
for epoch in range(num_epochs) :
    total_batch = mnist.train.num_examples / batch_size
    cost_avg = 0
    # for each batch for train data
    for batch in range(total_batch) :
        batch_X, batch_Y = mnist.train.next_batch(batch_size=batch_size)
        feed_dict = {X: batch_X, Y: batch_Y, keep_prob: 0.5}
        # optimize, and sum average cost
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        cost_avg += c / total_batch
    # print cost for current epoch
    print "Epoch: %04d cost= %.9f" % (epoch, cost_avg)

# correct prediction graph
correct = tf.cast(tf.equal(tf.arg_max(Y, 1), tf.arg_max(logits, 1)), tf.float32)
# accuracy graph
accuracy = tf.reduce_mean(correct)
# sess run for all test data
print ("Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))

import random
# one prediction test
# rand idx [0~test.num_exampels]
rand_idx = random.randint(0, mnist.test.num_examples-1)
#print lable for rand_idx
print "Label: ", sess.run(tf.arg_max(mnist.test.labels[rand_idx: rand_idx+1], 1))
#print prediction for rand_idx
print "Prediction: ", sess.run(tf.arg_max(logits,1), feed_dict={X:mnist.test.images[rand_idx: rand_idx+1], keep_prob: 1.0})
