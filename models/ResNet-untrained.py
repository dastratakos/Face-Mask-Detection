# Following as reference
# https://keras.io/guides/transfer_learning/#build-a-model

import numpy as np
import tensorflow as tf
from tensorflow import keras

data_augmentation = keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

IMG_HEIGHT = 64
IMG_WIDTH = 64

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
random_uniform_tensor = tf.keras.backend.random_uniform(shape=(IMG_HEIGHT, IMG_WIDTH, 3), minval=0.0, maxval=1.0)

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights=None,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Don't freeze base_model
base_model.trainable = True

# x = data_augmentation(inputs)  # optional data augmentation
x = inputs
x = tf.keras.applications.resnet.preprocess_input(x)  # ResNet50 input preprocessing
x = base_model(x, training=True)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(3)(x)
outputs = keras.layers.Activation('softmax')(x)
model = keras.Model(inputs, outputs)

print(model.summary())
print(model.predict(np.array([random_uniform_tensor])))