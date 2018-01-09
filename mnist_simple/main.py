# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Creating the model for our learning
    #
    # Basically, rather than looping over the images and doing the mathematical operations all at one time, we
    # go ahead and define placeholders for where each variable *will* go. We then define what the mathematical
    # operations *will* be run. We then send these placeholders and the definitions of the operations we want
    # to tensorflow and allow tensorflow to run these once it has all the data in one place on the hardware.
    #
    # It's basically analogous to defining a function or method that contains the instructions for what we want,
    # sending the function to the GPU or ML Cloud to be executed there all in one go before returning the result.

    # Each image is 28 by 28 pixels. (28 * 28) = 784. We're storing the image as a flat array, not multidimensional
    # Therefore it fits in a single 784 array
    images = tf.placeholder(tf.float32, [None, 784])
    weights = tf.Variable(tf.zeros([784, 10]))
    biases = tf.Variable(tf.zeros([10]))
    predicted_label = tf.matmul(images, weights) + biases

    # Define loss and optimizer
    #
    # This is also sometimes called "cost" or "correctness". It's basically where we'll store how far off from the
    # correct answer we were.
    correct_label = tf.placeholder(tf.float32, [None, 10])

    # Cross entropy is a way of "measuring how inefficient our predictions are for describing the truth"
    # Or in plainer terms, it's how we measure how wrong we are.
    #
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(correct_label * tf.log(tf.nn.softmax(predicted_label)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'predicted_label', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=predicted_label))

    # Back propagation. Also known as "Reverse-Mode Differentiation". This idea should be explained a bit more...
    # http://colah.github.io/posts/2015-08-Backprop/
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        session.run(train_step, feed_dict={images: batch_xs, correct_label: batch_ys})

    # Test trained model
    #
    # tf.argmax(predicted_label, 1) is what our model thinks is the most likely answer in the current set
    # tf.argmax(correct_label, 1) is what the *actual* correct answer is
    is_prediction_correct = tf.equal(tf.argmax(predicted_label, 1), tf.argmax(correct_label, 1))
    accuracy = tf.reduce_mean(tf.cast(is_prediction_correct, tf.float32))
    print(session.run(accuracy, feed_dict={images: mnist.test.images,
                                           correct_label: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
