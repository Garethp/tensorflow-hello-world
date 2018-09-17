from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')

# Import our data
(images_training, labels_training), (images_test, labels_test) = mnist.load_data()

# When we work with our data, we want to evaluate it as a one dimensional array, rather than the multiple (x, y, others)
# dimensions that images comes in. So we reshape our multi dimensional array in to a single dimensional array
num_pixels = images_training.shape[1] * images_training.shape[2]
images_training = images_training.reshape(images_training.shape[0], num_pixels).astype('float32')
images_test = images_test.reshape(images_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
images_training = images_training / 255
images_test = images_test / 255

# We're taking our labels (0, 1, 2, 3...) and turning them in to "categoricals". This way we can classify each of our
# dataset into the correct "category".
labels_training = np_utils.to_categorical(labels_training)
labels_test = np_utils.to_categorical(labels_test)
num_classes = labels_test.shape[1]

# Here we're going to create the model that we use to train the neural net
def baseline_model():

    model = Sequential()
    # We're going to create two "Dense" layers of neurons. A "Dense" layer is a layer where every node in that layer
    # is connected to every node in the previous layer. So every node in Layer 1 is connected to every node of our
    # inputs, and every node in Layer 2 is connected to every node in Layer 1.

    # For our first Layer, we're simply going to ask it to use the "ReLU" function on each node. ReLU is the most
    # widely used activation function for deep learning. Layer 1 has one node for every input
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))

    # Layer 2 is a layer with 10 nodes (the number of categories that we have), where we use the "Softmax" function on
    # the node output instead of ReLU. Softmax takes the values of all of the nodes and outputs them as a probability.
    # This means that each node will have a value between 0 and 1, and all nodes will add up to a value of 1.
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # We're going to compile the model, and ask it to calculate the "loss" (or how wrong we are) of our output layer
    # (The Layer 2 which is what number we think it is) against what the actual output should be using the
    # cross-entropy equation.

    # Once it measures the cross-entropy, we're going to apply the Stochastic Gradient Descent optomiser to try and
    # reduce our cross-entropy. Feel free to change this to 'adam' optomiser to see the differences. What optomimser
    # is best is something you play around with to try and find what fits your use case best.
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

model = baseline_model()

# Fit our data in to the model that we've created
model.fit(images_training, labels_training, validation_data=(images_test, labels_test), epochs=10, batch_size=200, verbose=2)

# Evaluate
scores = model.evaluate(images_test, labels_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
