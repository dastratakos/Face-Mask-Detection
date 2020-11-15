import tensorflow as tf

def loadDataset(img_directory):
    face_mask_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        img_directory, 
        labels='inferred', 
        class_names=["mask", "no_mask", "incorrect"],
        image_size=(64, 64),
        batch_size=1
    )
    return face_mask_dataset

def splitGroups(dataset, train_split, val_split, test_split):
    dataset_size = int(tf.data.experimental.cardinality(dataset))
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)

    dataset = dataset.shuffle(buffer_size=dataset_size)
    train_dataset = dataset.take(train_size)
    rest_dataset = dataset.skip(train_size)
    val_dataset = rest_dataset.take(val_size)
    test_dataset = rest_dataset.skip(val_size)

    print("Number of training samples: %d" % tf.data.experimental.cardinality(train_dataset))
    print("Number of validation samples: %d" % tf.data.experimental.cardinality(val_dataset))
    print("Number of test samples: %d" % tf.data.experimental.cardinality(test_dataset))

    return train_dataset, val_dataset, test_dataset