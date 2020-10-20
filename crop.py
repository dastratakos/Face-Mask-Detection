"""
file: crop.py
-------------
Takes each labeled input data set and creates new data. A new square image is
created for each person in the image. The corresponding labels (mask, no mask,
incorrect) are stored in a CSV file.

It should take just over 30 seconds to process 853 images.
"""
import csv
from PIL import Image

from tqdm import tqdm

from config import ARCHIVE_ROOT, IMAGE_ROOT, CROPPED_IMAGE_ROOT
import preprocess

CSV_FILE = f'{ARCHIVE_ROOT}cropped_labels.csv'
SCALE = 1.5
DIM = 64
NEW_LABELS = {
    'without_mask': 'no mask',
    'with_mask': 'mask',
    'mask_weared_incorrect': 'incorrect'
    }

def compute_crop_box(bound_box: dict):
    """Computes the coordinates to crop an image based on the provided
    bound_box and the SCALE variable. The input only describes a box around
    the mask (or area where the mask would be), and the width and height could
    could be different. The output box will be a square and contain a larger
    portion of the face.

    Args:
        bound_box (dict): Describes the original shape of the bounding
            box. The dimensions can be a rectangle (not a square).
    """
    x_max, x_min, y_max, y_min = [int(v) for _, v in sorted(bound_box.items())]

    # compute half of the side of the scaled box to crop    
    width = x_max - x_min
    height = y_max - y_min
    half_side = SCALE / 2 * max(width, height)

    # find the (x, y) center of the box
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    left, top, right, bottom = (int(x_center - half_side),
                                int(y_center - half_side),
                                int(x_center + half_side),
                                int(y_center + half_side))

    return((left, top, right, bottom))

def get_label(label: str) -> str:
    if label in NEW_LABELS:
        return NEW_LABELS[label]
    else:
        raise ValueError(f"Unknown label category: {label}")

def main():
    image_bases, annotations = preprocess.main()
    id = 0

    with open(CSV_FILE, 'w') as f:
        csv_file = csv.writer(f)
        csv_file.writerow(['id', 'label'])
        print(f'Finding faces in {len(image_bases)} images')
        with tqdm(total=len(image_bases)) as progress_bar:
            for image_base, annotation in zip(image_bases, annotations):
                im = Image.open(f'{IMAGE_ROOT}{image_base}')
                for object in annotation['objects']:
                    crop_box = compute_crop_box(object['bndbox'])
                    cropped_image = im.crop(crop_box).resize((DIM, DIM))
                    cropped_image.save(f'{CROPPED_IMAGE_ROOT}image{id}.png')
                    csv_file.writerow([id, get_label(object['name'])])
                    id += 1
                progress_bar.update()

if __name__ == '__main__':
    main()