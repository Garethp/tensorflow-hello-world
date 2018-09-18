# Classifying Dog Breeds

### Reading List
 * [The Original Article for this Code](http://machinememos.com/python/keras/artificial%20intelligence/machine%20learning/transfer%20learning/dog%20breed/neural%20networks/convolutional%20neural%20network/tensorflow/image%20classification/imagenet/2017/07/11/dog-breed-image-classification.html)
 * [Differences between ResNet and VGG](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)
 
 
### Network Architecture
```
  +-------+   +----------+
  | input |   |  input   |
  +-------+   +----------+
      |            |
      |            |
  +---v---+   +----v-----+
  | vgg19 |   | resnet50 |
  +-------+   +----------+
      |            |
      +------------+
            |
            v
    +--------------+
    | joined layer |
    +--------------+
            |
            v
       +---------+
       | dropout |
       +---------+
            |
            v
      +-----------+
      | normalize |
      +-----------+
            |
            v
       +---------+
       | dropout |
       +---------+
            |
            v
       +--------+
       | output |
       +--------+
```