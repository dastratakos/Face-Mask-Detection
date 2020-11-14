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

base_model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    classes=3
)

# Don't freeze base_model
base_model.trainable = True

# x = data_augmentation(inputs)  # optional data augmentation
x = inputs
x = tf.keras.applications.resnet.preprocess_input(x)  # ResNet50 input preprocessing
outputs = base_model(x, training=False)
model = keras.Model(inputs, outputs)

print(model.summary())