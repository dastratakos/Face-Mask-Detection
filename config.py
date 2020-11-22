"""
file: config.py
---------------
File containing high-level constants.
"""

ARCHIVE_ROOT = './archive/'
ANNOTATIONS_ROOT = ARCHIVE_ROOT + 'annotations/'
IMAGES_ROOT = ARCHIVE_ROOT + 'images/'
CROPPED_ROOT = ARCHIVE_ROOT + 'cropped/'
BALANCED_ROOT = ARCHIVE_ROOT + 'balanced/'
VISUALIZATION_ROOT = ARCHIVE_ROOT + 'visualizations/'
VISUALIZATION_TEST_ROOT = ARCHIVE_ROOT + 'visualizations_test/'
AUGMENTED_ROOT = ARCHIVE_ROOT + 'augmented/'
RESULTS_ROOT = './results/'

# logging format
FORMAT = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'

LABELS = {
    'no_mask': 0,
    'mask': 1,
    'incorrect': 2
    }