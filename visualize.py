"""
file: visualize.py
------------------
Visualizes images by drawing boxes to represent annotations. The color
of the box depends on the class of the annotation.

It should take around two and a half minutes to process 833 images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from tqdm import tqdm

from config import ARCHIVE_ROOT, IMAGE_ROOT, ANNOTATION_ROOT
import preprocess

VISUALIZATION_ROOT = f'{ARCHIVE_ROOT}visualizations/'
COLORS = {
    'with_mask': 'lime',
    'mask_weared_incorrect': 'yellow',
    'without_mask': 'red'
    }

def visualize_image(image_base: str, annotation: dict, interactive: bool=False):
    """ Creates a new image by drawing the bounding boxes from the annotation
    in one of three colors.

    Args:
        image_base (str): The base filename of the image
        annotation (dict): The corresponding annotation
        interactive (bool, optional): If true, the new image will be shown as a
            pop-up. Execution will be paused until the pop-up window is closed.
            Defaults to False.
    """
    image = np.array(Image.open(f'{IMAGE_ROOT}{image_base}'), dtype=np.uint8)
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

    id = int(image_base[12:-4])          # image id (i.e. maksssksksss[id].png)
    plt.title(f'Image {id}')
    if interactive: plt.show()
    else: fig.savefig(f'{VISUALIZATION_ROOT}image' + str(id) + '.png')
    plt.close()

if __name__ == '__main__':
    image_bases, annotations = preprocess.main()
    with tqdm(total=len(image_bases)) as progress_bar:
        for image_base, annotation in zip(image_bases, annotations):
            visualize_image(image_base, annotation)
            progress_bar.update()