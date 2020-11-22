"""
file: resNetUntrained.py
------------------------
Following as reference: https://keras.io/guides/transfer_learning/#build-a-model
"""
import logging

from keras_load_dataset import loadDataset, splitGroups
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras

from config import BALANCED_ROOT, FORMAT
from utils import util

IMG_HEIGHT = 64
IMG_WIDTH = 64

dataset_directory = BALANCED_ROOT
train_split = 0.8
val_split = 0.1
test_split = 0.1

def main():
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logging.info('========== ResNet Untrained ==========')

    face_mask_dataset = loadDataset(dataset_directory)
    train_set, val_set, test_set = splitGroups(face_mask_dataset, train_split, val_split, test_split)

    labels = np.array(np.concatenate([y for x, y in test_set], axis=0))
    print(labels)

    strategy = tf.distribute.MirroredStrategy()
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        data_augmentation = keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            ]
        )

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )

        # Don't freeze base_model
        base_model.trainable = True
        
        inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

        x = data_augmentation(inputs)  # optional data augmentation

        x = tf.keras.applications.resnet.preprocess_input(x)  # ResNet50 input preprocessing
        x = base_model(x, training=True)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(3)(x)
        outputs = keras.layers.Activation('softmax')(x)

        model = keras.Model(inputs, outputs)
        print(model.summary())

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    util.run_resnet_metrics('resnet50/untrained/', 'Before Training',
        model, test_set, labels)

    epochs = 20
    model.fit(train_set, epochs=epochs, validation_data=val_set)

    util.run_resnet_metrics('resnet50/untrained/', 'After Training',
        model, test_set, labels)

if __name__ == '__main__':
    main()