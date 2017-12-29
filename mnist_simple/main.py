import tensorflow
from mnist_simple import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Each image is 28 by 28 pixels. (28 * 28) = 784. We're storing the image as a flat array, not multidimensional
# Therefore it fits in a single 784 array
imagePlaceholder = tensorflow.placeholder(tensorflow.float32, [None, 784])

# How each pixels weighs against a number 0-9
weight = tensorflow.Variable(tensorflow.zeros([784, 10]))
bias = tensorflow.Variable(tensorflow.zeros([10]))

# y = softmax ((image * weight) + bias)
y = tensorflow.nn.softmax(tensorflow.matmul(imagePlaceholder, weight) + bias)

# This holds what the correct answer of an image should be
y_ = tensorflow.placeholder(tensorflow.float32, [None, 10])

# Cross entropy is a way of "measuring how inefficient our predictions are for describing the truth"
# Or in plainer terms, it's how we measure how wrong we are.
cross_entropy = tensorflow.reduce_mean(-tensorflow.reduce_sum(y_ * tensorflow.log(y), reduction_indices=[1]))

# Back propagation. Also known as "Reverse-Mode Differentiation". This idea should be explained a bit more...
# http://colah.github.io/posts/2015-08-Backprop/

# Use Example:
# c = a + b;
# d = b + 1;
# e = c * d;

# Using cross_entropy as a cost, change the weights and bias' in such a way that will move the cost closer to 0
train_step = tensorflow.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tensorflow.initialize_all_variables()
sess = tensorflow.Session()
sess.run(init)

# We're going to train over 5,000 images
for i in range(1000):
    # Get a random 100 points from the data set
    batchImages, batchYs = mnist.train.next_batch(100)
    # Run the train step, with the following variables
    sess.run(train_step, feed_dict={imagePlaceholder: batchImages, y_: batchYs})

# Check if our prediction matched the actual value
correct_prediction = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1))

# Turn the array of predictions ([True, True, False, True]) in to an array of floats [1, 1, 0, 1] then get the mean
# for that array (0.75). This is our accuracy
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

# Process the above function
print(sess.run(accuracy, feed_dict={imagePlaceholder: mnist.test.images, y_: mnist.test.labels}))
