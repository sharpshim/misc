#! /usr/bin/env python

import datetime

import data_helpers
import tensorflow as tf
import tensorflow.contrib.learn as learn
import numpy as np
from cnn import TextCNN
from tensorflow.python import debug as tf_debug
import time
import os

def class_statistics(y) :
    count_list = [0] * len(y[0])
    for yn in y:
        for i, ynn in enumerate(yn):
            if ynn == 1:
                count_list[i] += 1
    for i, c in enumerate(count_list):
        print "%d: %d" % (i, c)

tf.flags.DEFINE_string("news_data_file", "data/news.tsv", "news data")
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "dev sample percentage")
tf.flags.DEFINE_integer("embedding_size", 128, "embedding size")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "numb filters")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "l2 reg lambda")
tf.flags.DEFINE_float("keep_prob", 0.5, "keep prob")
tf.flags.DEFINE_integer("batch_size", 64, "bach size")
tf.flags.DEFINE_integer("num_epochs", 500, "num epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "evaluate every")
tf.flags.DEFINE_integer("checkpoint_every", 100, "checkpoint every")
tf.flags.DEFINE_integer("show_result_every", 1000, "show result every")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text, y = data_helpers.load_data_and_labels(FLAGS.news_data_file)
class_statistics(y)

max_document_length = max(len(x.split(" ")) for x in x_text)
# ["words in the sentence bla bla"] ==> [213, 12, 235, 546, 234, 0, 0, 0, 0]
vocb_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocb_processor.fit_transform(x_text)))
vocab_dict = vocb_processor.vocabulary_._mapping
#sorted(vocab_dict.items(), key = lambda x : x[1])
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
vocabulary = list(list(zip(*sorted_vocab))[0])


np.random.seed(100)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
x_text_shuffled = np.array(x_text)[shuffle_indices]

split_index = -1 * int(FLAGS.dev_sample_percentage * len(y))
x_train, x_dev, x_text_dev = x_shuffled[:split_index], x_shuffled[split_index:], x_text_shuffled[split_index:]
y_train, y_dev = y_shuffled[:split_index], y_shuffled[split_index:]

print "Vocabulary Size: {:d}".format(len(vocb_processor.vocabulary_))
print "Train/Dev: {:d}/{:d}".format(len(x_train), len(x_dev))

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    with sess.as_default():
        cnn = TextCNN(sequence_length = x_train.shape[1],
                      num_classes = y_train.shape[1],
                      vocb_size = len(vocb_processor.vocabulary_),
                      embedding_size = FLAGS.embedding_size,
                      filter_sizes = map(int, FLAGS.filter_sizes.split(",")),
                      num_filters = FLAGS.num_filters,
                      l2_reg_lambda = FLAGS.l2_reg_lambda
                      )
        global_step = tf.Variable(0, trainable=False, name="global_step")
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        vocb_processor.save(os.path.join(out_dir, "vocab"))

        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print "{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, x_text_batch, writer=None, show_result=False):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, prediction = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print "{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy)
            if writer:
                writer.add_summary(summaries, step)
            #print "prediction:", prediction
            if show_result:
                for i in range(len(x_batch)):
                    print "-------------------------------------------"
                    print x_batch[i]
                    print x_text_batch[i]
                    print y_batch[i]
                    print prediction[i]

        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            show_result = False
            if current_step % FLAGS.show_result_every == 0:
                show_result = True
            if current_step % FLAGS.evaluate_every == 0:
                print "\nEvaluation:"
                batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev, x_text_dev)), batch_size=1000, num_epochs=1)
                for batch_dev in batches_dev:
                    x_dev_batch, y_dev_batch, x_text_dev_batch = zip(*batch_dev)
                    dev_step(x_dev_batch, y_dev_batch, x_text_dev_batch, writer=dev_summary_writer, show_result = show_result)

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

writer = tf.summary.FileWriter("./graph", graph=graph)
writer.close()
