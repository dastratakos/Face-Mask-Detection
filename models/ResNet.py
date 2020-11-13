# Following as reference
# https://keras.io/guides/transfer_learning/#build-a-model

import numpy as np
import tensorflow as tf
from tensorflow import keras

# data_augmentation = keras.Sequential(
#     [
#         layers.experimental.preprocessing.RandomFlip("horizontal"),
#         layers.experimental.preprocessing.RandomRotation(0.1),
#     ]
# )

IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_DEPTH = 64

base_model = tf.keras.applications.ResNet50(
    include_top=False,  # don't include fully-connected layer on top so we can build and train our own
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)
)

# Freeze base_model
base_model.trainable = False

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
# x = data_augmentation(inputs)