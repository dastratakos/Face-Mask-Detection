"""
file: visualize.py
------------------
Visualizes images by drawing boxes to represent annotations. The color
of the box depends on the class of the annotation.

It should take around two and a half minutes to process 853 images.
"""
import argparse
import logging
import os

import cv2
from tqdm import tqdm

from config import IMAGES_ROOT, ANNOTATIONS_ROOT, VISUALIZATION_ROOT, \
    VISUALIZATION_TEST_ROOT, FORMAT
from utils import util

# in BGR representation for OpenCV
COLORS = {
    'with_mask': (0, 255, 0), # green
    'mask_weared_incorrect': (0, 255, 255), # yellow
    'without_mask': (0, 0, 255) # red
    }

def visualize_image(file_in_name: str, file_out_name: str, annotation: dict):
    """
    Creates a new image by drawing the bounding boxes from the annotation in
    one of three colors.

    Args:
        file_in_name (str): Full path to the image
        file_out_name (str): Full path of where to save the visualized image
        annotation (dict): Represents the annotation of the input image,
            including the bounding boxes
    """
    im = cv2.imread(file_in_name)

    for object in annotation['objects']:
        box = [int(v) for k, v in sorted(object['bndbox'].items())]
        x_max, x_min, y_max, y_min = box
        cv2.rectangle(im, (x_min, y_min), (x_max, y_max), COLORS[object['name']], 1)

    cv2.imwrite(file_out_name, im)

def main():
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    print()
    logging.info('========== Visualizing bounding boxes ==========')
    os.makedirs(VISUALIZATION_ROOT, exist_ok=True)

    for i in tqdm(range(util.get_num_images())):
        file_in_name = IMAGES_ROOT + f'maksssksksss{i}.png'
        file_out_name = VISUALIZATION_ROOT + f'image-{i}.png'
        annotation = util.parseXML(
            ANNOTATIONS_ROOT + f'maksssksksss{i}.xml')
        visualize_image(file_in_name, file_out_name, annotation)
    
    print()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description=util.build_description('Image visualization module'),
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-t', '--test',
        help='visualize a single image',
        action='store_true')
    arg_parser.add_argument('-i', '--image_id',
        help='id of the image to visualize',
        type=int,
        default=0)
    args = arg_parser.parse_args()

    if args.test:
        os.makedirs(VISUALIZATION_TEST_ROOT, exist_ok=True)
        file_in_name = IMAGES_ROOT + f'maksssksksss{args.image_id}.png'
        file_out_name = VISUALIZATION_TEST_ROOT + f'image-{args.image_id}.png'
        annotation = util.parseXML(
            ANNOTATIONS_ROOT + f'maksssksksss{args.image_id}.xml')
        visualize_image(file_in_name, file_out_name, annotation)
    else: main()