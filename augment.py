"""
file: augment.py
----------------
Increases the size of the dataset by performing data augmentation. The factor
by which the dataset increases is specified by the FACTOR variable.

This should take around 1 minute per additional factor for 833 images.

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

TODO:
    - create new label CSV
    - investigate low contrast images
"""
import os
from os import pread
import random

import numpy as np
from scipy import ndarray
from scipy.ndimage import uniform_filter
from skimage import io
from skimage.transform import rotate, warp, AffineTransform
from skimage.util import random_noise, img_as_ubyte
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.exposure import adjust_gamma, adjust_log, adjust_sigmoid
from tqdm import tqdm

from config import ARCHIVE_ROOT, CROPPED_IMAGE_ROOT, AUGMENTED_IMAGE_ROOT

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
    return img_as_ubyte(rgb2gray(image))

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
    # flip_vertical,# NOTE: Does it make sense to flip a face upside-down?
    # gray_scale,   # NOTE: Makes each [R,G,B] into a single element (not compatible)
    change_contrast,
    gamma_correction,
    log_correction,
    sigmoid_correction,
    # blur,         # NOTE: The images are already pretty blury at 64x64
    shear
]

def test_all_transformations():
    AUGMENTED_IMAGE_TEST_ROOT = f'{ARCHIVE_ROOT}augmented_test/'
    os.makedirs(AUGMENTED_IMAGE_TEST_ROOT, exist_ok=True)

    image_base = 'image1.png'
    augmented_base = f'{AUGMENTED_IMAGE_TEST_ROOT}{image_base[:-4]}'
    
    im = io.imread(f'{CROPPED_IMAGE_ROOT}{image_base}')
    io.imsave(f'{augmented_base}-original.png', im)
    for transformation in TRANSFORMATIONS:
        im2 = transformation(im)
        name = transformation.__name__
        io.imsave(f'{augmented_base}-{name}.png', im2)

def main():
    os.makedirs(f'{ARCHIVE_ROOT}augmented', exist_ok=True)

    # sort by the image id (i.e. image[id].png)
    image_bases = list(sorted(os.listdir(CROPPED_IMAGE_ROOT),
                              key=lambda x: int(x[5:-4])))
    with tqdm(total=len(image_bases)) as progress_bar:
        for image_base in image_bases:
            augmented_base = f'{AUGMENTED_IMAGE_ROOT}{image_base[:-4]}'
            im = io.imread(f'{CROPPED_IMAGE_ROOT}{image_base}')
            io.imsave(f'{augmented_base}-0.png', im)
            for i in range(1, FACTOR):
                num_transformations = random.randint(1, len(TRANSFORMATIONS))
                for _ in range(num_transformations):
                    transform = random.choice(TRANSFORMATIONS)
                    im = transform(im)
                io.imsave(f'{augmented_base}-{i}.png', im)
            progress_bar.update()

if __name__ == '__main__':
    main()
    # test_all_transformations()