import os
import csv
import collections

import numpy as np
import torch as tor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARCHIVE_ROOT, CROPPED_IMAGE_ROOT, AUGMENTED_IMAGE_ROOT
from simpleimage import SimpleImage

RGB_GRANULARITY = 32
TOTAL_RGB = 256
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20

RANDOM_DATA_NUM_POINTS = 10

def FeatureExtractor(data):
    num_cats = TOTAL_RGB // RGB_GRANULARITY
    output = np.zeros((num_cats, num_cats, num_cats))
    for pixel in data:
        red = pixel[0] // RGB_GRANULARITY
        green = pixel[1] // RGB_GRANULARITY
        blue = pixel[2] // RGB_GRANULARITY
        output[red][green][blue] += 1
    return output.flatten()

def CalculateLogRegGrad(theta, features, y, n):
    def g(t, dp):
        return (1 + np.exp(-np.dot(t.T, dp))) ** -1
    return (y - g(theta, features)) * features / n

def LearnPredictor(x_points, y_points, FE, numIters, eta, eps):
    thetas = np.zeros((TOTAL_RGB//RGB_GRANULARITY)**3)
    for t in range(numIters):
        for i in range(len(x_points)):
            featureVector = FE(x_points[i])
            gradient = CalculateLogRegGrad(thetas, featureVector, y_points[i], len(y_points))
            prev_thetas = thetas.copy()
            thetas = thetas + eta * gradient
            if np.linalg.norm(thetas - prev_thetas) < eps:
                return thetas
    return thetas

# def GenerateRandomData(numPoints, imageWidth, imageHeight):
#     x_points = tor.randint(TOTAL_RGB, (numPoints, imageWidth * imageHeight, 3))
#     y_points = tor.randint(2, (numPoints,))
#     return np.array(x_points), np.array(y_points)

def Classify(validation_x_points, thetas):
    predictions = np.zeros(len(validation_x_points))
    for i in range(len(predictions)):
        predictions[i] = 1 / (1 + np.exp(-np.dot(thetas.T, FeatureExtractor(validation_x_points[i]))))
    return predictions

# x_points, y_points = GenerateRandomData(RANDOM_DATA_NUM_POINTS, IMAGE_WIDTH, IMAGE_HEIGHT)
# thetas = LearnPredictor(x_points, y_points, FeatureExtractor, 10000, 0.02, 1e-5)
# val_x_points, val_y_points = GenerateRandomData(RANDOM_DATA_NUM_POINTS, IMAGE_WIDTH, IMAGE_HEIGHT)
# preds = Classify(val_x_points, thetas)
# correct = 0
# for i in range(len(preds)):
#     if preds[i] >= 0.5:
#         if val_y_points[i] == 1:
#             correct += 1
#     else:
#         if val_y_points[i] == 0:
#             correct += 1
# print("Num correct: " + str(correct/len(preds)))

def LogReg():
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
        if label[1] == "no mask": y.append(0)
        elif label[1] == "mask": y.append(1)
        else: y.append(2)

    y = y[:num_examples]

    index = 0
    for image_base in image_bases:
        image = SimpleImage(CROPPED_IMAGE_ROOT + image_base)
        X.append([((pixel.red + pixel.green + pixel.blue) // 3) for pixel in image])
        index += 1
        if index >= num_examples: break

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100000))
    clf.fit(X_train, y_train)
    print("Logistic Regression Score is", clf.score(X_test, y_test))
    print("Logistic Regression Balanced Score is", balanced_accuracy_score(y_test, clf.predict(X_test)))