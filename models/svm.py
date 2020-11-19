"""
file: svm.py
------------
Implements a Support Vector Machine using various kernels.
"""
import logging

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import ARCHIVE_ROOT, BALANCED_IMAGE_ROOT, CROPPED_IMAGE_ROOT, \
                   AUGMENTED_IMAGE_ROOT, FORMAT
from utils import util

KERNELS = {
    'linear',
    'rbf',
    'poly'
}

def main(csv_path:      str  = ARCHIVE_ROOT + 'augmented_labels.csv',
         image_root:    str  = AUGMENTED_IMAGE_ROOT,
         balanced_root: str  = BALANCED_IMAGE_ROOT,
         use_balanced:  bool = True,
         interactive:   bool = False):
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logging.info('========== Support Vector Machine ==========')

    if use_balanced:
        X_train, X_test, y_train, y_test = util.load_dataset_from_split(balanced_root)
    else:
        X_train, X_test, y_train, y_test = util.load_dataset(csv_path, image_root)

    for kernel in KERNELS:
        logging.info(f'Kernel type: {kernel}')
        clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, probability=True))
        logging.info('\tFitting the model...')
        clf.fit(X_train, y_train)

        util.run_metrics(clf, X_test, y_test, f'svm/{kernel}/',
                         f'SVM ({kernel})', interactive)

if __name__ == '__main__':
    main(ARCHIVE_ROOT + 'cropped_labels.csv', CROPPED_IMAGE_ROOT)