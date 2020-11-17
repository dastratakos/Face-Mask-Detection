"""
file: config.py
---------------
This file includes all of the high-level, hard-coded information.
TODO: extend this file to be modified based on the command line arguments.
"""

ARCHIVE_ROOT = './archive/'
ANNOTATION_ROOT = ARCHIVE_ROOT + 'annotations/'
IMAGE_ROOT = ARCHIVE_ROOT + 'images/'
CROPPED_IMAGE_ROOT = ARCHIVE_ROOT + 'cropped/'
AUGMENTED_IMAGE_ROOT = ARCHIVE_ROOT + 'augmented/'
CROPPED_BY_CLASS_ROOT = ARCHIVE_ROOT + 'cropped_by_class/'
BALANCED_IMAGE_ROOT = ARCHIVE_ROOT + 'balanced/'
NUM_CLASSES_IN_MODEL = 3

# logging format
FORMAT = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'

# verbose = False
# etc...