import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.datasets.mnist as mnist

#import tensorflow as tf
from tensorflow.python.eager import context

_ = tf.Variable([1])

context._context = None
context._create_context()

tf.config.threading.set_inter_op_parallelism_threads(1)


#num_threads=10
#tf.config.threading.set_inter_op_parallelism_threads( num_threads )

from timeit import default_timer as timer
import numpy as np

import threading

# Preparação para CNN
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
x_train = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
x_test = test_images.astype('float32') / 255

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

start = timer()

new_model = tf.keras.models.load_model('MnistModel.h5')
new_model.summary()
res=new_model.predict(train_images)
#print(res.shape)

end = timer()
print('tempo total: ',end - start)

