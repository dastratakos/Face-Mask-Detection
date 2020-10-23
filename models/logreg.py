"""
file: logreg.py
------------
Implements Logistic Regression.
"""
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARCHIVE_ROOT, CROPPED_IMAGE_ROOT, AUGMENTED_IMAGE_ROOT, FORMAT
from utils import util

def main(csv_path: str = ARCHIVE_ROOT + 'augmented_labels.csv',
         image_root: str = AUGMENTED_IMAGE_ROOT):
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logging.info('========== Logistic Regression ==========')

    X_train, X_test, y_train, y_test = util.load_dataset(csv_path, image_root)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100000))
    logging.info('Fitting the model...')
    clf.fit(X_train, y_train)
    logging.info(f'score: {clf.score(X_test, y_test)}')
    logging.info(f'balanced score: {balanced_accuracy_score(y_test, clf.predict(X_test))}')

if __name__ == '__main__':
    main(ARCHIVE_ROOT + 'cropped_labels.csv', CROPPED_IMAGE_ROOT)