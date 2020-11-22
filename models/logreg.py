"""
file: logreg.py
------------
Implements Logistic Regression.
"""
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARCHIVE_ROOT, BALANCED_ROOT, CROPPED_ROOT, FORMAT
from utils import util

def main(balanced_root: str = BALANCED_ROOT, interactive: bool = False):
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    print()
    logging.info('========== Logistic Regression ==========')

    X_train, X_test, y_train, y_test = util.load_dataset(balanced_root)
    
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100000))
    logging.info('Fitting the model...')
    clf.fit(X_train, y_train)
    
    util.run_baseline_metrics(clf, X_test, y_test, 'logreg/',
        'Logistic Regression', interactive)

if __name__ == '__main__':
    main(ARCHIVE_ROOT + 'cropped_labels.csv', CROPPED_ROOT)