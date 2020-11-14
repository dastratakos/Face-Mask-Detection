import tensorflow as tf

def loadDataset(img_directory):
    face_mask_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        img_directory, 
        labels='inferred', 
        class_names=["mask", "no_mask", "incorrect"],
        image_size=(64, 64)
    )
    return face_mask_dataset

def splitGroups(dataset, train_split):
    dataset_size = tf.data.experimental.cardinality(dataset)
    train_size = int(train_split * dataset_size)

    dataset = dataset.shuffle()
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    print("Number of training samples: %d" % tf.data.experimental.cardinality(train_dataset))
    print("Number of test samples: %d" % tf.data.experimental.cardinality(test_dataset))

    return train_dataset, test_dataset