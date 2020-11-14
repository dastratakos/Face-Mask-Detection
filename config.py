"""
file: config.py
---------------
This file includes all of the high-level, hard-coded information.
TODO: extend this file to be modified based on the command line arguments.
"""

ARCHIVE_ROOT = './archive/'
IMAGE_ROOT = ARCHIVE_ROOT + 'images/'
ANNOTATION_ROOT = ARCHIVE_ROOT + 'annotations/'
CROPPED_IMAGE_ROOT = ARCHIVE_ROOT + 'cropped/'
AUGMENTED_IMAGE_ROOT = ARCHIVE_ROOT + 'augmented/'
CROPPED_CLASS_ROOT = ARCHIVE_ROOT + 'images_classes/'
NUM_CLASSES_IN_MODEL = 3

# verbose = False
# etc...

def build_description(title: str) -> str:
    lines = [
        title,
        '',
        'Charles Pan, Gilbert Rosal, & Dean Stratakos',
        'CS 229: Machine Learning',
        'October 20, 2020'
    ]

    max_length = max(len(l) for l in lines)
    bar = f"+{'-' * (max_length + 2)}+"
    description = f'{bar}\n'
    for line in lines:
        total_padding = max_length - len(line)
        left_padding = int(total_padding / 2)
        right_padding = int((total_padding + 1) / 2)
        description += f"| {' ' * left_padding}{line}{' ' * right_padding} |\n"
    description += f'{bar}'

    return description