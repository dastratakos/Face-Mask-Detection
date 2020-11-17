# Following as reference
# https://keras.io/guides/transfer_learning/#build-a-model

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras_load_dataset import loadDataset, splitGroups
from sklearn import metrics

dataset_directory = "./archive/balanced"
train_split = 0.8
val_split = 0.1
test_split = 0.1

face_mask_dataset = loadDataset(dataset_directory)
train_set, val_set, test_set = splitGroups(face_mask_dataset, train_split, val_split, test_split)

labels = np.concatenate([y for x, y in test_set], axis=0)
print(labels)

IMG_HEIGHT = 64
IMG_WIDTH = 64

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

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

print("BEFORE TRAINING EVALUATION")
print("MODEL EVALUATION")
print(model.evaluate(test_set))
print("BALANCED ACCURACY")
print(tf.argmax(input=model.predict(test_set), axis=1).numpy())
print(metrics.balanced_accuracy_score(labels, model.predict(test_set)))

# epochs = 20
# model.fit(train_set, epochs=epochs, validation_data=val_set)

# print("AFTER FINE TUNING EVALUATION")
# print("MODEL EVALUATION")
# print(model.evaluate(test_set))
# print("BALANCED ACCURACY")
# print(metrics.balanced_accuracy_score(labels, model.predict(test_set)))

# with strategy.scope():
#     # fine tune over the whole model
#     base_model.trainable = True
#     model.compile(
#         optimizer=keras.optimizers.Adam(1e-5),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
#     )
    
# epochs = 10
# model.fit(train_set, epochs=epochs, validation_data=val_set)

# print("AFTER TOTAL MODEL TRAINING EVALUATION")
# print("MODEL EVALUATION")
# print(model.evaluate(test_set))
# print("BALANCED ACCURACY")
# print(metrics.balanced_accuracy_score(labels, model.predict(test_set)))