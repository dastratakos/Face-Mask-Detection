import tensorflow as tf

def loadDataset(img_directory):
    face_mask_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        img_directory, 
        labels='inferred', 
        class_names=["mask", "no_mask", "incorrect"],
        image_size=(64, 64)
    )
    return face_mask_dataset

def splitGroups(dataset):
    dataset_size = tf.data.experimental.cardinality(dataset)

    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = int(0.1 * dataset_size)

    dataset = dataset.shuffle()
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)

    print("Number of training samples: %d" % tf.data.experimental.cardinality(train_dataset))
    print("Number of validation samples: %d" % tf.data.experimental.cardinality(val_dataset))
    print("Number of test samples: %d" % tf.data.experimental.cardinality(test_dataset))

    return train_dataset, val_dataset, test_dataset