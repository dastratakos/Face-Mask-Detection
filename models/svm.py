"""
file: svm.py
------------
Implements a Support Vector Machine using various kernels.
"""
import logging

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import ARCHIVE_ROOT, BALANCED_ROOT, CROPPED_ROOT, FORMAT
from utils import util

KERNELS = {
    'linear',
    'rbf',
    'poly'
}

def main(balanced_root: str = BALANCED_ROOT, interactive: bool = False):
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    print()
    logging.info('========== Support Vector Machine ==========')

    X_train, X_test, y_train, y_test = util.load_dataset(balanced_root)

    for kernel in KERNELS:
        logging.info(f'Kernel type: {kernel}')
        clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, probability=True))
        logging.info('\tFitting the model...')
        clf.fit(X_train, y_train)

        util.run_baseline_metrics(clf, X_test, y_test, f'svm/{kernel}/',
            f'SVM ({kernel})', interactive)
    
    print()

if __name__ == '__main__':
    main(ARCHIVE_ROOT + 'cropped_labels.csv', CROPPED_ROOT)