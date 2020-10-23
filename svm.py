import os
import csv
import collections

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

from config import ARCHIVE_ROOT, CROPPED_IMAGE_ROOT, AUGMENTED_IMAGE_ROOT
from simpleimage import SimpleImage

def SVM():
    X = []
    labels = []

    image_bases = list(sorted(os.listdir(CROPPED_IMAGE_ROOT),
                                key=lambda x: int(x[5:-4])))

    with open(ARCHIVE_ROOT + 'cropped_labels.csv') as f:
        labels = [line for line in csv.reader(f)][1:]
        labels = [[int(line[0]), line[1]] for line in labels]
        print(f'y has {len(labels)} elements')

    num_examples = len(labels)
    y = []
    for label in labels:
        if label[1] == "no mask": 
            y.append(0)
        elif label[1] == "mask": 
            y.append(1)
        else: 
            y.append(2)

    y = y[:num_examples]

    index = 0
    for image_base in image_bases:
        image = SimpleImage(CROPPED_IMAGE_ROOT + image_base)
        X.append([((pixel.red + pixel.green + pixel.blue) // 3) for pixel in image])
        index += 1
        if index >= num_examples: break

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    clf = make_pipeline(StandardScaler(), SVC(kernel = "linear"))
    clf.fit(X_train, y_train)
    print("Linear Kernel Score is", clf.score(X_test, y_test))
    print("Linear Kernel Balanced Score is", balanced_accuracy_score(y_test, clf.predict(X_test)))

    clf = make_pipeline(StandardScaler(), SVC(kernel = "rbf"))
    clf.fit(X_train, y_train)
    print("RBF Kernel Score is", clf.score(X_test, y_test))
    print("RBF Kernel Balanced Score is", balanced_accuracy_score(y_test, clf.predict(X_test)))

    clf = make_pipeline(StandardScaler(), SVC(kernel = "poly"))
    clf.fit(X_train, y_train)
    print("Polynomial Kernel Score is", clf.score(X_test, y_test))
    print("Polynomial Kernel Balanced Score is", balanced_accuracy_score(y_test, clf.predict(X_test)))