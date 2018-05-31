
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}
    # the first layer takes 1 input and has 32 5x5 filters --> giving us 32 outputs
    # the second layer takes 32 inputs, has 64 5x5 filters --> giving us 64 outputs
    # the third layer is 7*7*64 as each image is downsized to be 7x7 and there are 64 of them, and 1024 nodes

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}
    # the dimension of the bias always matches the number of outputs

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.conv2d(x, weights['W_conv1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(hm_epochs):
        epoch_loss = 0

        for _ in range(int(mnist.train.num_examples/batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            # the _, converts c to be a float32 from a list as it says ignore the first [optimizer] value
            epoch_loss += c


        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    with sess.as_default():
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
