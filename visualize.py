"""
file: visualize.py
------------------
Visualizes images by drawing boxes to represent annotations. The color
of the box depends on the class of the annotation.
"""


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from preprocess import parseXML

COLORS = {
    'with_mask': 'lime',
    'mask_weared_incorrect': 'yellow',
    'without_mask': 'red'
    }

def visualize_image(image_index, annotation, interactive=False):
    image_filename = './archive/images/maksssksksss' + str(image_index) + '.png'
    image = np.array(Image.open(image_filename), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for object in annotation['objects']:
        x_max, x_min, y_max, y_min = [int(v) for k, v in sorted(object['bndbox'].items())]
        rect = patches.Rectangle(
            (x_min, y_min),
            (x_max - x_min),
            (y_max - y_min),
            linewidth=1,
            edgecolor=COLORS[object['name']],
            facecolor='none')
        ax.add_patch(rect)

    plt.title(f'Image {image_index}')
    if interactive: plt.show()
    else: fig.savefig('./archive/visualizations/maksssksksss' + str(image_index) + '.png')
    plt.close()

if __name__ == '__main__':
    for index in range(257, 833):
        visualize_image(index, parseXML('./archive/annotations/maksssksksss' + str(index) + '.xml'))