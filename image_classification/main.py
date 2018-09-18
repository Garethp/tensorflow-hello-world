import numpy
import keras
import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import os.path

import matplotlib.pyplot as plt

(images_training, labels_training), (images_test_import, labels_test_import) = cifar10.load_data()

num_classes = 10
labels_training = keras.utils.to_categorical(labels_training, num_classes)
labels_test = keras.utils.to_categorical(labels_test_import, num_classes)

images_training = images_training.astype('float32')
images_test = images_test_import.astype('float32')
images_training /= 255
images_test /= 255

model_filename = 'model.h5'

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=images_training.shape[1:]))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # opt = keras.optimizers.adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if not os.path.isfile(model_filename):
    model = createModel()
    batch_size = 256
    epochs = 100
    history = model.fit(images_training, labels_training, batch_size=32, epochs=100, verbose=2, validation_data=(images_test, labels_test), shuffle=True)
    model.save(model_filename)
else:
    model = keras.models.load_model(model_filename)

scores = model.evaluate(images_test, labels_test, verbose=0)

print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

predictions = []
for x in range(5):
    image_to_show = random.randint(0, images_test.shape[0])
    predicted = model.predict(numpy.expand_dims(images_test[image_to_show], axis=0))
    predictions.append(predicted)
    plt.title('Predicted: ' + labels[numpy.argmax(predicted)])
    plt.imshow(images_test_import[image_to_show])
    plt.show()
