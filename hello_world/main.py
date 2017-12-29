import tensorflow

hello = tensorflow.constant('Hello, Tensorflow!')
session = tensorflow.Session()
print(session.run(hello))