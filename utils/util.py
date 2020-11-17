"""
file: util.py
-------------
Contains utility functions.
"""

import logging
import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_image_bases(image_root: str) -> list:
    """ Returns a list of sorted image_bases contained in the image_root. The
    image_bases are sorted by the image id, then face id, then augment id.
    (i.e. image-[image id]-[face id]-[augment id].png) """
    return list(sorted(os.listdir(image_root), key=lambda x: tuple(
        int(x.split('.')[0].split('-')[i]) for i in range(1, len(x.split('-'))))))

def load_dataset(csv_path: str, image_root: str):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         image_root: Root folder of the images corresponding to the CSV file.

    Returns:
        X_train, X_test, y_train, y_test
    """

    image_bases = get_image_bases(image_root)

    logging.info(f'Loading {len(image_bases)} images...')
    X = []
    for image_base in tqdm(image_bases):
        """ Load the image as a numpy array. Then, reshape it to be a vector
        of dimension equal to the total number of pixels, while converting it
        to grayscale by taking the average the RGB values. Index slicing is
        used to remove the 4th pixel value (alpha) if needed. """
        im = np.array(Image.open(image_root + image_base))[:,:,:3]
        im = np.average(im.reshape(im.shape[0]**2, 3), axis=1)
        X.append(im)

    y = [int(x) for x in np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=[-1])]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

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