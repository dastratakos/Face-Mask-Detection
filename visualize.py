"""
file: visualize.py
------------------
Visualizes images by drawing boxes to represent annotations. The color
of the box depends on the class of the annotation.

It should take around two and a half minutes to process 853 images.
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from tqdm import tqdm

from config import ARCHIVE_ROOT, IMAGE_ROOT, ANNOTATION_ROOT, build_description
import preprocess

VISUALIZATION_ROOT = ARCHIVE_ROOT + 'visualizations/'
COLORS = {
    'with_mask': 'lime',
    'mask_weared_incorrect': 'yellow',
    'without_mask': 'red'
    }

def visualize_image(image_id: int, interactive: bool=False):
    """ Creates a new image by drawing the bounding boxes from the annotation
    in one of three colors.

    Args:
        image_id (int): The id of the image to visualize
        interactive (bool, optional): If true, the new image will be shown as a
            pop-up. Execution will be paused until the pop-up window is closed.
            Defaults to False.
    """
    image = np.array(
        Image.open(IMAGE_ROOT + f'maksssksksss{image_id}.png'),
        dtype=np.uint8)
    annotation = preprocess.parseXML(
        ANNOTATION_ROOT + f'maksssksksss{image_id}.xml')
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for object in annotation['objects']:
        x_max, x_min, y_max, y_min = [int(v)
                                for k, v in sorted(object['bndbox'].items())]
        rect = patches.Rectangle(
            (x_min, y_min),
            (x_max - x_min),
            (y_max - y_min),
            linewidth=1,
            edgecolor=COLORS[object['name']],
            facecolor='none')
        ax.add_patch(rect)

    plt.title(f'Image {image_id}')
    if interactive: plt.show()
    else: fig.savefig(VISUALIZATION_ROOT + f'image-{image_id}.png')
    plt.close()

def main():
    os.makedirs(ARCHIVE_ROOT + 'visualizations/', exist_ok=True)

    for index in tqdm(range(preprocess.get_num_images())):
        visualize_image(index)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description=build_description('Image visualization module'),
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument("-t", "--test",
        help="visualize a single image",
        action="store_true")
    arg_parser.add_argument("-i", "--image_id",
        help="id of the image to visualize",
        type=int,
        default=0)
    args = arg_parser.parse_args()

    if args.test: visualize_image(args.image_id, interactive=True)
    else: main()