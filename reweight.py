"""
file: reweight.py
-----------------
Upsample minority classes using data augmentation to create new data points
so that each class has an equal number of data points.
"""
import argparse
import csv
import logging
import os
import random
import shutil

import numpy as np
from PIL import Image
from skimage import io
from tqdm import tqdm

from config import ARCHIVE_ROOT, BALANCED_IMAGE_ROOT, IMAGE_ROOT, CROPPED_IMAGE_ROOT, FORMAT
from data_processing import augment
from data_processing import preprocess
from utils import util

CSV_FILE = ARCHIVE_ROOT + 'cropped_labels.csv'
TARGET = 2000           # new num no mask and num mask
TARGET_MINORITY = 500   # new num incorrect

def upsample_no_mask():
    print("upsampling no mask")
    filenames = os.listdir('archive/images_classes/no_mask')
    # copy original images
    for filename in filenames:
        shutil.copy('archive/images_classes/no_mask/' + filename, 'archive/balanced/no_mask')
    # augment the rest
    for i in tqdm(range(TARGET - len(filenames))):
        filename = random.choice(filenames)
        im = io.imread('archive/images_classes/no_mask/' + filename)
        im2 = augment.augment_image(im)
        io.imsave(f"archive/balanced/no_mask/{filename.split('.')[0]}-{i}.png", im2)

def upsample_incorrect():
    print("upsampling incorrect")
    filenames = os.listdir('archive/images_classes/incorrect')
    # copy original images
    for filename in filenames:
        shutil.copy('archive/images_classes/incorrect/' + filename, 'archive/balanced/incorrect')
    # augment the rest
    for i in tqdm(range(TARGET_MINORITY - len(filenames))):
        filename = random.choice(filenames)
        im = io.imread('archive/images_classes/incorrect/' + filename)
        im2 = augment.augment_image(im)
        io.imsave(f'archive/balanced/incorrect/{filename}-{i}.png', im2)

def downsample_mask(cropped):
    print("downsampling mask")
    filenames = os.listdir('archive/images_classes/mask')
    downsampled = random.sample(filenames, TARGET)
    for filename in tqdm(downsampled):
        shutil.copy('archive/images_classes/mask/' + filename, 'archive/balanced/mask')

def main():
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logging.info('========== Reweighting module ==========')
    os.makedirs(ARCHIVE_ROOT + 'balanced', exist_ok=True)

    # image_bases, annotations = preprocess.main()

    cropped = np.loadtxt(CSV_FILE, delimiter=',', skiprows=1)
    total = cropped.shape[0]

    num_no_mask = len(os.listdir('archive/images_classes/no_mask'))
    num_mask = len(os.listdir('archive/images_classes/mask'))
    num_incorrect = len(os.listdir('archive/images_classes/incorrect'))
    print(f'num_no_mask:    {num_no_mask} \t> {num_no_mask / total : .2%}')
    print(f'num_mask:       {num_mask} \t> {num_mask / total : .2%}')
    print(f'num_incorrect:  {num_incorrect} \t> {num_incorrect / total : .2%}')
    print(f'total:          {total}')

    upsample_no_mask()
    upsample_incorrect()
    downsample_mask(cropped)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description=util.build_description('Reweighting module'),
        formatter_class=argparse.RawTextHelpFormatter)

    main()