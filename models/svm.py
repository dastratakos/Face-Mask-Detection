"""
file: svm.py
------------
Implements a Support Vector Machine using various kernels.
"""
import logging

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

from config import ARCHIVE_ROOT, CROPPED_IMAGE_ROOT, AUGMENTED_IMAGE_ROOT, FORMAT
from utils import util

KERNELS = {
    'linear',
    'rbf',
    'poly'
}

def main(csv_path: str = ARCHIVE_ROOT + 'augmented_labels.csv',
         image_root: str = AUGMENTED_IMAGE_ROOT):
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logging.info('========== Support Vector Machine ==========')

    X_train, X_test, y_train, y_test = util.load_dataset(csv_path, image_root)

    for kernel in KERNELS:
        logging.info(f'Kernel type: {kernel}')
        clf = make_pipeline(StandardScaler(), SVC(kernel=kernel))
        logging.info('\tFitting the model...')
        clf.fit(X_train, y_train)
        logging.info(f'\tscore: {clf.score(X_test, y_test)}')
        logging.info(f'\tbalanced score: {balanced_accuracy_score(y_test, clf.predict(X_test))}')

if __name__ == '__main__':
    main(ARCHIVE_ROOT + 'cropped_labels.csv', CROPPED_IMAGE_ROOT)