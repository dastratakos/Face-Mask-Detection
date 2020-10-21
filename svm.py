import os
import csv
import collections

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import ARCHIVE_ROOT, CROPPED_IMAGE_ROOT
from simpleimage import SimpleImage

X = []
labels = []

image_bases = list(sorted(os.listdir(CROPPED_IMAGE_ROOT),
                            key=lambda x: int(x[5:-4])))

with open(ARCHIVE_ROOT + 'cropped_labels.csv') as f:
    labels = [line for line in csv.reader(f)][1:]
    labels = [[int(line[0]), line[1]] for line in labels]
    print(f'y has {len(labels)} elements')
    print(f'y[0] = {labels[0]}')

num_examples = 4072
y = []
index = 0
num_mask = 0
num_no_mask = 0
num_incorrect = 0
for label in labels:
    if label[1] == "no mask": 
        y.append(0)
        num_no_mask += 1
    elif label[1] == "mask": 
        y.append(1)
        num_mask += 1
    else: 
        y.append(2)
        num_incorrect += 1
print(f"Num correctly wearing mask: {num_mask / num_examples} Num with no mask: {num_no_mask / num_examples} Num incorrect: {num_incorrect / num_examples}")


y = y[:num_examples]

index = 0
for image_base in image_bases:
    image = SimpleImage(CROPPED_IMAGE_ROOT + image_base)
    X.append([((pixel.red + pixel.green + pixel.blue) // 3) for pixel in image])
    index += 1
    if index >= num_examples: break

# # placeholder data
# # x_train = [[0, 0], [1, 1]]
# # y_train = [0, 1]
# # x_val = [[2, 2]]
# # y_val = [[1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# X_train = [[0, 0, 0], [1, 1, 1]]
# Y_train = [0, 1]

# X_val = [[-1, -1, -1]]
# Y_val = [1]

linearclf = make_pipeline(StandardScaler(), SVC())
linearclf.fit(X_train, y_train)

predictions = linearclf.predict(X_test)
print(predictions)
print("Score is", linearclf.score(X_test, y_test))