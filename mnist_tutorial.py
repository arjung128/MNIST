from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/tensorflow/mnist/input_data/MNIST_data")#, one_hot=True)

# FLAGS = None

def main(_):

  # print(len(mnist.train.images))    # 55,000
  # print(len(mnist.train.labels))    # 55,000
  # print(len(mnist.train.images[0])) # 784 = 28 x 28
  # print(mnist.train.images.shape)   # [55000, 784] -- 55,000 rows, 784 columns
  # print(mnist.train.labels.shape)   # [55000, ] -- 55,000 rows, 1 column
  # print(mnist.train.labels[0])      # 7
  # print(mnist.train.labels[1])      # 3
  # print(mnist.train.labels[2])      # 4
  # print(mnist.train.images[0])      # An array with 784 numbers
  # print(mnist.train.images[0].shape)# [784, ] -- 784 rows, 1 column
  # print(mnist.train.images[0:2])   # A matrix with two arrays of 784 numbers
  # print(mnist.train.labels[0:2])   # An array with 2 numbers

  # Why is there a need for a separate validation dataset?
  # Does softmax give a y-output for each pixel? That doesn't seem right...

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])   # None for variable batch_size
  W = tf.Variable(tf.zeros([784, 10]))  # 10 so that output is 55,000 x 10 for one_hot
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # tf.Variable for trainable variables such as weights (W) and biases (B) for your model.
  # tf.placeholder is used to feed actual training examples.

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None]) # Label, i.e. Correct answer

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    acc_train = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # print(acc_train)

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(
      accuracy, feed_dict={
          x: mnist.test.images,
          y_: mnist.test.labels
      }))


# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument(
#       '--data_dir',
#       type=str,
#       default='/tmp/tensorflow/mnist/input_data',
#       help='Directory for storing input data')
#   FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main)#, argv=[sys.argv[0]] + unparsed)

# https://stats.stackexchange.com/questions/308777/why-are-there-no-deep-reinforcement-learning-engines-for-chess-similar-to-alpha
# https://arxiv.org/pdf/1509.01549.pdf
# https://www.technologyreview.com/s/541276/deep-learning-machine-teaches-itself-chess-in-72-hours-plays-at-international-master/
# https://stackoverflow.com/questions/753954/how-to-program-a-neural-network-for-chess
