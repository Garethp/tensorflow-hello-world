from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

from keras.preprocessing import image
from tqdm import tqdm

from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# We want to load our dataset from actual images and files, as opposed from a dataset provided by Keras
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# We need to be able to convert the images that we've downloaded into arrays
def convert_image_to_tensors(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def convert_images_to_tensors(img_paths):
    list_of_tensors = [convert_image_to_tensors(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# Takes a list of images and converts them in to input tensors then inserts them into a pre-trained
# VGG19 network using weights from training on the ImageNet dataset
def get_VGG19_network_with_images(file_paths):
    tensors = convert_images_to_tensors(file_paths).astype('float32')
    preprocessed_input = preprocess_input_vgg19(tensors)
    return VGG19(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

# Takes a list of images and converts them in to input tensors then inserts them into a pre-trained
# ResNet50 network using weights from training on the ImageNet dataset
def get_resnet50_network_with_images(file_paths):
    tensors = convert_images_to_tensors(file_paths).astype('float32')
    preprocessed_input = preprocess_input_resnet50(tensors)
    return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

# Creates a Keras "branch" with the defined layers. It then returns both the input layer of teh branch and the end
# of the branch. This way we can define what information goes in to the input later on
def get_input_branch(input_shape=None):
    size = int(input_shape[2] / 4)

    branch_input = Input(shape=input_shape)
    branch = GlobalAveragePooling2D()(branch_input)
    branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)

    # Batch Normalization is attempting to normalize the outputs of the previous layers (in this case, the output of
    # the networks we're plugging in) so that drastic changes in values of the previous layers have a smaller effect on
    # the later layers. Since every node and every layer is training at the same time, a small shift in a weight in the
    # first layers can become a medium shift in the second or third layer and result in a massive shift in the layers
    # towards the bottom. By normalizing at various intervals, we can make sure that small shifts early on don't cause
    # a large ripple effect, allowing the later layers to train in a more stable independent way. Essentially, without
    # this a small shift in earlier layers can completely mess up the training values in later layers, which would take
    # more and more training to fix. By stabilising the input values to later layers and allowing their training to be
    # more independent, we end up speeding over all training time.
    #
    # It also has a nice side effect of introducing some mathematical noise due to how it plays out, which can assist in
    # preventing some overfitting when used with dropout layers. It's not there to replace dropout layers, but it can
    # help
    branch = BatchNormalization()(branch)
    branch = Activation("relu")(branch)
    return branch, branch_input

train_files, train_targets = load_dataset('assets/images/train')
valid_files, valid_targets = load_dataset('assets/images/valid')
test_files, test_targets = load_dataset('assets/images/test')

dog_names = [item[20:-1] for item in sorted(glob("dog/assets/images/train/*/"))]

# Fetch and configure our VGG19 network
train_vgg19 = get_VGG19_network_with_images(train_files)
valid_vgg19 = get_VGG19_network_with_images(valid_files)
test_vgg19 = get_VGG19_network_with_images(test_files)
print("VGG19 shape", train_vgg19.shape[1:])

# Fetch and configure our ResNet50 network
train_resnet50 = get_resnet50_network_with_images(train_files)
valid_resnet50 = get_resnet50_network_with_images(valid_files)
test_resnet50 = get_resnet50_network_with_images(test_files)
print("Resnet50 shape", train_resnet50.shape[1:])

# We're creating our two input branches for the two networks before merging them together. Here's what our overall
# architecture should look like in the end
#
#  +-------+   +----------+
#  | Input |   |  Input   |
#  +-------+   +----------+
#      |            |
#      |            |
#  +---v---+   +----v-----+
#  | VGG19 |   | ResNet50 |
#  +-------+   +----------+
#      |            |
#      +------------+
#            |
#            v
#    +--------------+
#    | Joined Layer |
#    +--------------+
#            |
#            v
#       +---------+
#       | Dropout |
#       +---------+
#            |
#            v
#      +-----------+
#      | Normalize |
#      +-----------+
#            |
#            v
#       +---------+
#       | Dropout |
#       +---------+
#            |
#            v
#       +--------+
#       | Output |
#       +--------+

vgg19_branch, vgg19_input = get_input_branch(input_shape=(7, 7, 512))
resnet50_branch, resnet50_input = get_input_branch(input_shape=(7, 7, 2048))
concatenate_branches = Concatenate()([vgg19_branch, resnet50_branch])

net = Dropout(0.3)(concatenate_branches)
net = Dense(640, use_bias=False, kernel_initializer='uniform')(net)
net = BatchNormalization()(net)
net = Activation("relu")(net)
net = Dropout(0.3)(net)
net = Dense(133, kernel_initializer='uniform', activation="softmax")(net)

model = Model(inputs=[vgg19_input, resnet50_input], outputs=[net])

# Train
model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='saved_models/bestmodel.hdf5',
                               verbose=1, save_best_only=True)
model.fit([train_vgg19, train_resnet50], train_targets,
          validation_data=([valid_vgg19, valid_resnet50], valid_targets),
          epochs=30, batch_size=4, callbacks=[checkpointer], verbose=1)
