"""
file: preprocess.py
-------------------
Parses XML files.
"""
import argparse
import copy
import logging
import os
import pprint
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from config import IMAGE_ROOT, ANNOTATION_ROOT, CROPPED_IMAGE_ROOT, \
    NUM_CLASSES_IN_MODEL, CROPPED_BY_CLASS_ROOT, ARCHIVE_ROOT, FORMAT, LABELS
from utils import util

def separateByClass():
    """
    Using the cropped images, this function creates a new directory where the
    images are separated by their class (Mask, No mask, Incorrect).
    """
    os.makedirs(CROPPED_BY_CLASS_ROOT, exist_ok=True)
    for label in LABELS:
        os.makedirs(CROPPED_BY_CLASS_ROOT + label, exist_ok=True)
    labels = np.loadtxt(ARCHIVE_ROOT + 'cropped_labels.csv', delimiter=',',
                        skiprows=1)
    labels = [list(x) for x in labels]
    filenames = util.get_image_bases(CROPPED_IMAGE_ROOT)
    for filename in filenames:
        image_id = [int(x) for x in filename.split('.')[0].split('-')[1:]]
        for i, label in enumerate(list(LABELS.keys())):
            temp = copy.deepcopy(image_id)
            temp.append(i)
            if temp in labels:
                shutil.copy(CROPPED_IMAGE_ROOT + filename,
                            CROPPED_BY_CLASS_ROOT + label)

def createImageClassesFolder():
    """
    From the CROPPED_IMAGE_ROOT, this function generates a set of directories
    that contain all the images for masked people, unmasked people, and people
    wearing masks incorrect.
    :return: A folder with n subfolders, where n is the number of classes in
    the ML model and each subfolder contains all the images for the given class.
    """
    if not os.path.isdir(CROPPED_BY_CLASS_ROOT): os.mkdir(CROPPED_BY_CLASS_ROOT)
    for class_num in range(NUM_CLASSES_IN_MODEL):
        path = CROPPED_BY_CLASS_ROOT + "class_" + str(class_num) + '/'
        if not os.path.isdir(path): os.mkdir(path)
    labels = pd.read_csv(ARCHIVE_ROOT + 'cropped_labels.csv')
    files = util.get_image_bases(CROPPED_IMAGE_ROOT)
    for i in range(len(files)):
        image_num = int(files[i][files[i].find('e') + 1: files[i].find('.')])
        if labels['label'][image_num] == 'mask':
            shutil.copy(CROPPED_IMAGE_ROOT + files[i], CROPPED_BY_CLASS_ROOT + 'class_0')
        elif labels['label'][image_num] == 'no mask':
            shutil.copy(CROPPED_IMAGE_ROOT + files[i], CROPPED_BY_CLASS_ROOT + 'class_1')
        else:
            shutil.copy(CROPPED_IMAGE_ROOT + files[i], CROPPED_BY_CLASS_ROOT + 'class_2')

def parseXML(xml_filename: str) -> dict:
    """ This function generates an annotation dictionary representation of
    the contents of the specified XML filename.

    Args:
        xml_filename (str): Relative path to the XML file

    Returns:
        annotation (dict): Fepresentation of the entire XML file
    """
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    annotation = {'objects': []}
    for item in root.findall('./'):
        if item.tag == 'size':
            annotation['size'] = {dim.tag:dim.text for dim in item}
        elif item.tag == 'object':
            annotation['objects'].append(
                {child.tag:(child.text if child.tag != 'bndbox'
                    else {c.tag:c.text for c in child})
                for child in item})
        else:
            annotation[item.tag] = item.text             
    return annotation

def single_file_test(xml_base):
    """ Tests the parseXML function for a single XML file.
    """
    annotation = parseXML(ANNOTATION_ROOT + xml_base)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(annotation)

def get_num_images() -> int:
    return len(os.listdir(IMAGE_ROOT))

def main():
    """ Collects all of the images and annotations from the archive directory.
    Converts the annotation XML files to a list of dictionaries.

    Returns:
        image_bases ([str]): List of the filenames for all images
        annotations ([dict]): List of the parsed annotations
    """
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logging.info('========== Preprocessing Module ==========')

    # sort by the image id (i.e. maksssksksss[image id].png)
    image_bases = list(sorted(os.listdir(IMAGE_ROOT),
                              key=lambda x: int(x[12:-4])))
    annotations = list(sorted(os.listdir(ANNOTATION_ROOT),
                              key=lambda x: int(x[12:-4])))
    assert len(image_bases) == len(annotations), \
        f'Number of images ({len(image_bases)}) does not match the number of \
        annotations ({len(annotations)})'
    
    logging.info(f'The dataset contains {len(image_bases)} data points')
    annotations = [parseXML(ANNOTATION_ROOT + annotation)
                    for annotation in annotations]

    return image_bases, annotations

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description=util.build_description('Preprocess module'),
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-t', '--test',
        help='parse a single XML file',
        action='store_true')
    arg_parser.add_argument('-f', '--file',
        help='XML file name to run tests on',
        default='maksssksksss0.xml')
    args = arg_parser.parse_args()

    if args.test: single_file_test(args.file)
    else: main()
    
