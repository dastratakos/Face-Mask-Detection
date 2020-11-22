"""
file: balance.py
----------------
This module combats an issue we were facing with the distribution of our
dataset classes. The original breakdown of images is as follows:

No mask: 	717 	 17.61%
Mask: 		3232 	 79.37%
Incorrect: 	123 	 3.02%
--------------------------------
Total: 		4072

Using this module, we upsample the minority classes, No mask and Incorrect, and
downsample the majority class, Mask. A new directory is created with the new
dataset, which includes 1000 images for Incorrect and 2000 images for each of
No mask and Mask.
"""
import argparse
import logging
import os
import random
import shutil

from skimage import io
from tqdm import tqdm

from config import BALANCED_ROOT, CROPPED_ROOT, FORMAT
from utils import augment, util

TARGET_MASK = 2000          # new number for Mask
TARGET_NO_MASK = 2000       # new number for No Mask
TARGET_INCORRECT = 1000     # new number for Incorrect

def upsample(className, target):
    os.makedirs(BALANCED_ROOT + className, exist_ok=True)
    filenames = os.listdir(CROPPED_ROOT + className)
    print(f'Upsampling {className} from {len(filenames)} to {target}...')
    with tqdm(total=target) as progress_bar:
        # copy original images
        for filename in filenames:
            shutil.copy(CROPPED_ROOT + f'{className}/' + filename,
                        BALANCED_ROOT + className)
            progress_bar.update()
        # augment the rest
        for i in range(target - len(filenames)):
            filename = random.choice(filenames)
            im = io.imread(CROPPED_ROOT + f'{className}/' + filename)
            im2 = augment.augment_image(im)
            io.imsave(BALANCED_ROOT + \
                      f"{className}/{filename.split('.')[0]}-{i}.png", im2)
            progress_bar.update()

def downsample(className, target):
    os.makedirs(BALANCED_ROOT + className, exist_ok=True)
    filenames = os.listdir(CROPPED_ROOT + className)
    print(f'Downsampling {className} from {len(filenames)} to {target}...')
    downsampled = random.sample(filenames, target)
    for filename in tqdm(downsampled):
        shutil.copy(CROPPED_ROOT + f'{className}/' + filename,
                    BALANCED_ROOT + className)

def printDistribution():
    num_no_mask = len(os.listdir(CROPPED_ROOT + 'no_mask'))
    num_mask = len(os.listdir(CROPPED_ROOT + 'mask'))
    num_incorrect = len(os.listdir(CROPPED_ROOT + 'incorrect'))
    total = num_no_mask + num_mask + num_incorrect
    print()
    print(f'No mask: \t{num_no_mask} \t{num_no_mask / total : .2%}')
    print(f'Mask: \t\t{num_mask} \t{num_mask / total : .2%}')
    print(f'Incorrect: \t{num_incorrect} \t{num_incorrect / total : .2%}')
    print('-' * 32)
    print(f'Total: \t\t{total}')
    print()

def main():
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    print()
    logging.info('========== Balancing the dataset ==========')
    os.makedirs(BALANCED_ROOT, exist_ok=True)

    printDistribution()

    upsample('no_mask', TARGET_NO_MASK)
    upsample('incorrect', TARGET_INCORRECT)
    downsample('mask', TARGET_MASK)

    print()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description=util.build_description('Balancing module'),
        formatter_class=argparse.RawTextHelpFormatter)

    main()