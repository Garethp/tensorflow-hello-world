import numpy
import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model

import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')

# Import our data
(images_training, labels_training), (images_test_import, labels_test) = mnist.load_data()

# Flatten our image data in to a single dimensional array
num_pixels = images_training.shape[1] * images_training.shape[2]
images_training = images_training.reshape(images_training.shape[0], 1, 28, 28).astype('float32')
images_test = images_test_import.reshape(images_test_import.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
images_training = images_training / 255
images_test = images_test / 255

# Convert the labels in to categoricals
labels_training = np_utils.to_categorical(labels_training)
labels_test = np_utils.to_categorical(labels_test)
num_classes = labels_test.shape[1]

def baseline_model():
    model = Sequential()

    # We're creating a convolution layer here to create 32 feature maps from our image.
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))

    # We're pooling those convolution layers so that we can reduce the total number of nodes in our network.
    # If we don't do this, we can easily end up with many more nodes than we need per image, taking up more
    # resources and time than needed.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Let's add a dropout layer. Dropout layers allow us to randomly drop nodes from our neural network
    # in order to prevent overfitting
    model.add(Dropout(0.2))

    # Our Max Pooling returns a two dimensional array of nodes. We want to flatten that down to a single dimension
    # so that we can more easily connect it to a dense layer
    model.add(Flatten())

    # Creating our fully connected layer to connect to the flattened pooling. We should have roughly one node
    # per node outputted by the max pooling, so that the NN can apply weights to them individually
    model.add(Dense(512, activation='relu'))

    # Create our output layer, where we try to classify the image
    model.add(Dense(num_classes, activation='softmax'))

    # Use the cross-entropy equation for our loss calculator and apply an optomiser to that loss calculation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create our model
model = baseline_model()
#
# # Fit our data in to the model we've created
model.fit(images_training, labels_training, validation_data=(images_test, labels_test), epochs=5, batch_size=200, verbose=2)
#
# # Final evaluation of the model
scores = model.evaluate(images_test, labels_test, verbose=0)
model.save('model.h5')
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

predictions = []
for x in range(5):
    image_to_show = random.randint(0, images_test.shape[0])
    predicted = model.predict(numpy.expand_dims(images_test[image_to_show], axis=0))
    predictions.append(predicted)
    plt.title('Predicted: ' + str(numpy.argmax(predicted)))
    plt.imshow(images_test_import[image_to_show])
    plt.show()
