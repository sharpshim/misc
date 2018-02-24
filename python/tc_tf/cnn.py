import tensorflow as tf

class TextCNN:
    def __init__(self, embedding_size, filter_sizes, num_filters,
                 sequence_length, vocb_size, num_classes, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], "input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, None, "keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocb_size, embedding_size], -1, -1), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            print self.embedded_chars
            self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)
            print self.embedded_chars_expand

        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%d" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name="W")
                #W = tf.Variable(tf.random_uniform(shape=filter_shape, minval=-1, maxval=-1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expand, W, [1, 1, 1, 1], "VALID", name="conv")
                relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(relu, [1, sequence_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pooled")
                pooled_outputs.append(pooled)
        num_filters_total = len(filter_sizes) * num_filters
        print pooled_outputs
        self.h_pool = tf.concat(pooled_outputs, 3)
        print self.h_pool
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print self.h_pool_flat

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob, name="dropout")

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            #W = tf.Variable(tf.random_uniform([num_filters_total, num_classes], -1, -1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop,
                                          W, b, name="scores")
            self.predictions = tf.arg_max(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
