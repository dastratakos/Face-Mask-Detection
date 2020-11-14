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

IMG_HEIGHT = 150
IMG_WIDTH = 150

base_model = tf.keras.applications.ResNet50(
    include_top=False,  # don't include fully-connected layer on top so we can build and train our own
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
)

# Freeze base_model
base_model.trainable = False

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)  # optional data augmentation

x = tf.keras.applications.resnet.preprocess_input(x)  # ResNet50 input preprocessing

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(3)(x)
outputs = keras.layers.Activation('softmax')(x)

model = keras.Model(inputs, outputs)

print(model.summary())