"""
file: logreg.py
------------
Implements Logistic Regression.
"""
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARCHIVE_ROOT, BALANCED_IMAGE_ROOT, CROPPED_IMAGE_ROOT, \
                   AUGMENTED_IMAGE_ROOT, FORMAT
from utils import util

def main(csv_path:      str  = ARCHIVE_ROOT + 'augmented_labels.csv',
         image_root:    str  = AUGMENTED_IMAGE_ROOT,
         balanced_root: str  = BALANCED_IMAGE_ROOT,
         use_balanced:  bool = True,
         interactive:   bool = False):
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logging.info('========== Logistic Regression ==========')

    if use_balanced:
        X_train, X_test, y_train, y_test = util.load_dataset_from_split(balanced_root)
    else:
        X_train, X_test, y_train, y_test = util.load_dataset(csv_path, image_root)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100000))
    logging.info('Fitting the model...')
    clf.fit(X_train, y_train)
    
    util.run_metrics(clf, X_test, y_test, 'logreg/', 'Logistic Regression', interactive)

if __name__ == '__main__':
    main(ARCHIVE_ROOT + 'cropped_labels.csv', CROPPED_IMAGE_ROOT)