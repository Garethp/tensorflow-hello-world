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

# flatten 28*28 images to a 784 vector for each image
num_pixels = images_training.shape[1] * images_training.shape[2]
images_training = images_training.reshape(images_training.shape[0], 1, 28, 28).astype('float32')
images_test = images_test_import.reshape(images_test_import.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
images_training = images_training / 255
images_test = images_test / 255

# one hot encode outputs
labels_training = np_utils.to_categorical(labels_training)
labels_test = np_utils.to_categorical(labels_test)
num_classes = labels_test.shape[1]

def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
model.fit(images_training, labels_training, validation_data=(images_test, labels_test), epochs=5, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(images_test, labels_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

predictions = []
for x in range(5):
    image_to_show = random.randint(0, images_test.shape[0])
    predicted = model.predict(numpy.expand_dims(images_test[image_to_show], axis=0))
    predictions.append(predicted)
    plt.title('Predicted: ' + str(numpy.argmax(predicted)))
    plt.imshow(images_test_import[image_to_show])
    plt.show()
