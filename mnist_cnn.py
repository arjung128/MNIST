import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
print(x.shape)
# dynamically reshape the input
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
print(x_shaped.shape)

batch_size = 50
batch_x = mnist.train.next_batch(batch_size=batch_size)
print(len(batch_x[2]))
