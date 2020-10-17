import numpy as np
import torch as tor
import collections

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
    prev_thetas = []
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


def GenerateRandomData(numPoints, imageWidth, imageHeight):
    x_points = tor.randint(TOTAL_RGB, (numPoints, imageWidth * imageHeight, 3))
    y_points = tor.randint(2, (numPoints,))
    return np.array(x_points), np.array(y_points)

def Classify(validation_x_points, thetas):
    predictions = np.zeros(len(validation_x_points))
    for i in range(len(predictions)):
        predictions[i] = 1 / (1 + np.exp(-np.dot(thetas.T, FeatureExtractor(validation_x_points[i]))))
    return predictions

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x_points, y_points = GenerateRandomData(RANDOM_DATA_NUM_POINTS, IMAGE_WIDTH, IMAGE_HEIGHT)
    thetas = LearnPredictor(x_points, y_points, FeatureExtractor, 10000, 0.02, 1e-5)
    val_x_points, val_y_points = GenerateRandomData(RANDOM_DATA_NUM_POINTS, IMAGE_WIDTH, IMAGE_HEIGHT)
    preds = Classify(val_x_points, thetas)
    correct = 0
    for i in range(len(preds)):
        if preds[i] >= 0.5:
            if val_y_points[i] == 1:
                correct += 1
        else:
            if val_y_points[i] == 0:
                correct += 1
    print("Num correct: " + str(correct/len(preds)))