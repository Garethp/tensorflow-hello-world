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

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# Creates a deep neural network for processing our images. It does this by
#  * Reshaping the image tensor
#  * Running the image tensor through a convolution to produce 32 features
#  * Pooling (max) the features from that convolution
#  * Running the pooled features through another convolution to produce 64 features
#  * Pooling (max) the second convolution
#  * Flatening the produced downsampled features
#  * Creating a fully connected layer to the downsampled features
#  * Running the fully connected layer through a dropout
#  * Returning a readout layer of tensors connected to the tensors kept during dropout.
def create_deep_neural_net(x):
    """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_probability). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_probability is a scalar placeholder for the probability of
    dropout.
  """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # greyscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one greyscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        convolution1_bias = bias_variable([32])
        convolution1 = tf.nn.relu(get_2d_convolution(x_image, W_conv1) + convolution1_bias)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        pool1 = get_max_pool(convolution1)

    # Second convolutional layer -- maps the 32 features outputted by the first convolution to 64 new features
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        convolution2_bias = bias_variable([64])
        convolution2 = tf.nn.relu(get_2d_convolution(pool1, W_conv2) + convolution2_bias)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        pool2 = get_max_pool(convolution2)

    # Fully connected layer 1 -- after 2 rounds of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        fully_connected_layer_weight = weight_variable([7 * 7 * 64, 1024])
        fully_connected_layer_bias = bias_variable([1024])

        # Flatten the pooled tensors in a single array
        pool2_flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])
        fully_connected_layer = tf.nn.relu(tf.matmul(pool2_flattened, fully_connected_layer_weight) + fully_connected_layer_bias)

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('dropout'):
        keep_probability = tf.placeholder(tf.float32)
        fully_connected_layer_dropped = tf.nn.dropout(fully_connected_layer, keep_probability)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        readout_weight = weight_variable([1024, 10])
        readout_bias = bias_variable([10])

        y_conv = tf.matmul(fully_connected_layer_dropped, readout_weight) + readout_bias
    return y_conv, keep_probability

# Performs a 2D Convolution over the input with the given filter.
# Args:
#   input: A tensor with the shape [batch, input_height, input_width, input_channels]
#   filter: A 4D tensor with the shape [filter_height, filter_width, input_channels, output_channels]
def get_2d_convolution(input, filter):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# Performs a max pool operation with a 2x2 filter on the given input tensor
def get_max_pool(input):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Create a weighted variable with a standard deviation of 0.1. We do this because we use ReLU Neurons, which basically
# just apply an max(0, x) to an input variable so that we don't have negative values. Because of this, we want to have
# positive starting biases so that we don't end up with a lot of neurons defaulting to 0.
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the tensor placeholder for an image
    image_placeholder = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    correct_answer = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = create_deep_neural_net(image_placeholder)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=correct_answer,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        is_prediction_correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(correct_answer, 1))
        is_prediction_correct = tf.cast(is_prediction_correct, tf.float32)
    accuracy = tf.reduce_mean(is_prediction_correct)

    # For performance, we'll save the graph that we've built
    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 20,000 Steps of Training
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    image_placeholder: batch[0], correct_answer: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={image_placeholder: batch[0], correct_answer: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            image_placeholder: mnist.test.images, correct_answer: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
