"""
file: augment.py
----------------
Increases the size of the dataset by performing data augmentation. The factor
by which the dataset increases is specified by the FACTOR variable.

This should take around 1 minute per additional factor for 853 images.

Reference:
    https://www.kaggle.com/tomahim/image-manipulation-augmentation-with-skimage

Operations include:
    - rotating
    - adding noise
    - flipping (horizontal)
    - flipping (vertical)
    - gray scaling
    - changing contrast
    - gamma correction
    - log correction
    - sigmoid correction
    - blur
    - shear

Other operations that are not immediately relevant to the specific application.
Namely, if the images are always centered and scaled similarly, then scaling
and translation are not important. Furthermore, the model will not need to
train on images with inverted colors.
    - scale in
    - scale out
    - translation
    - color inversion

NOTE: There is a risk of creating low contrast images if too many
transformations are applied, or if gamma correction is used.
"""
import argparse
import csv
import logging
import os
import random

import numpy as np
from scipy import ndarray
from scipy.ndimage import uniform_filter
from skimage import io
from skimage.transform import rotate, warp, AffineTransform
from skimage.util import random_noise, img_as_ubyte
from skimage.color import rgb2gray, rgba2rgb
from skimage.exposure import rescale_intensity
from skimage.exposure import adjust_gamma, adjust_log, adjust_sigmoid
from tqdm import tqdm

from config import ARCHIVE_ROOT, CROPPED_IMAGE_ROOT, AUGMENTED_IMAGE_ROOT, FORMAT
from utils import util

CSV_FILE = ARCHIVE_ROOT + 'augmented_labels.csv'
FACTOR = 3

def random_rotation(image: ndarray):
    return img_as_ubyte(rotate(image, random.uniform(-30, 30)))

def add_noise(image: ndarray):
    return img_as_ubyte(random_noise(image))

def flip_horizontal(image: ndarray):
    return image[:, ::-1]

def flip_vertical(image: ndarray):
    return image[::-1, :]

def gray_scale(image: ndarray):
    return img_as_ubyte(rgb2gray(rgba2rgb(image)))

def change_contrast(image: ndarray):
    v_min, v_max = np.percentile(image, (0.2, 99.8))
    return rescale_intensity(image, in_range=(v_min, v_max))

def gamma_correction(image: ndarray):
    return adjust_gamma(image, gamma=0.4, gain=0.9)

def log_correction(image: ndarray):
    return adjust_log(image)

def sigmoid_correction(image: ndarray):
    return adjust_sigmoid(image)

def blur(image: ndarray):
    return uniform_filter(image, size=(11, 11, 1))

def shear(image: ndarray):
    return img_as_ubyte(warp(image, inverse_map=AffineTransform(shear=0.2)))

TRANSFORMATIONS = [
    random_rotation,
    add_noise,
    flip_horizontal,
    change_contrast,
    log_correction,
    shear
]
UNUSED = [
    flip_vertical,        # NOTE: Irrelevant to flip a face upside-down
    gray_scale,           # NOTE: Reduces each [R,G,B] into a single number
    gamma_correction,     # NOTE: Makes the image too bright
    sigmoid_correction,   # NOTE: Makes the image too dark
    blur                  # NOTE: The images are already a bit blury at 64x64
]

def test_all_transformations(image_base):
    AUGMENTED_IMAGE_TEST_ROOT = ARCHIVE_ROOT + 'augmented_test/'
    os.makedirs(AUGMENTED_IMAGE_TEST_ROOT, exist_ok=True)

    augmented_base = AUGMENTED_IMAGE_TEST_ROOT + image_base[:-4]

    im = io.imread(CROPPED_IMAGE_ROOT + image_base)
    io.imsave(f'{augmented_base}-original.png', im)
    
    all_transformations = TRANSFORMATIONS + UNUSED
    logging.info(f'Applying {len(all_transformations)} transformations')
    for transformation in tqdm(all_transformations):
        im2 = transformation(im)
        name = transformation.__name__
        io.imsave(f'{augmented_base}-{name}.png', im2)

def augment_image(im, num_transformations=2):
    im2 = im
    for _ in range(num_transformations):
        transform = random.choice(TRANSFORMATIONS)
        im2 = transform(im2)
    return im2


def main():
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logging.info('========== Augmenting Module ==========')
    os.makedirs(ARCHIVE_ROOT + 'augmented', exist_ok=True)

    # labels = {'[image id]-[face id]': label for each line}
    with open(ARCHIVE_ROOT + 'cropped_labels.csv') as f:
        labels = {f'{line[0]}-{line[1]}': int(line[2])
                    for line in csv.reader(f) if line[0].isnumeric()}

    image_bases = util.get_image_bases(CROPPED_IMAGE_ROOT)

    with open(CSV_FILE, 'w') as f:
        csv_file = csv.writer(f)
        csv_file.writerow(['image id', 'face_id', 'augment_id', 'label'])
        logging.info(f'Augmenting {len(image_bases)} images')
        with tqdm(total=len(image_bases)) as progress_bar:
            for image_base in image_bases:
                image_id, face_id = (int(image_base[:-4].split('-')[1]),
                                     int(image_base[:-4].split('-')[2]))
                label = labels[f'{image_id}-{face_id}']
                augmented_base = AUGMENTED_IMAGE_ROOT + image_base[:-4]

                im = io.imread(CROPPED_IMAGE_ROOT + image_base)
                io.imsave(f'{augmented_base}-0.png', im)
                csv_file.writerow([image_id, face_id, 0, label])
                for augment_id in range(1, FACTOR):
                    num_transformations = random.randint(1, 3)
                    im2 = augment_image(im, num_transformations)
                    io.imsave(f'{augmented_base}-{augment_id}.png', im2)
                    csv_file.writerow([image_id, face_id, augment_id, label])
                progress_bar.update()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description=util.build_description('Data augmentation module'),
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-t', '--test',
        help='run a test of all transformations',
        action='store_true')
    arg_parser.add_argument('-f', '--file',
        help='image file name to run tests on',
        default='image-22-0.png')
    args = arg_parser.parse_args()

    if args.test: test_all_transformations(args.file)
    else: main()