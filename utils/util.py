"""
file: util.py
-------------
Contains utility functions.
"""
from datetime import datetime
import json
import logging
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import scikitplot as skplt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sn
import tensorflow as tf
from tqdm import tqdm

from config import ANNOTATIONS_ROOT, IMAGES_ROOT, RESULTS_ROOT, LABELS

def create_confusion_matrix():
    # confusion matrix for pretrained ResNet50
    array = [[81,76,32],[78,77,42],[50,46,18]]

    df_cm = pd.DataFrame(array, range(3), range(3))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})

    plt.title('Pretrained ResNet50 Confusion Matrix')
    plt.savefig(RESULTS_ROOT + 'resnet50/pretrained/confusion_matrix.png')

def run_resnet_metrics(dir_name, note, model, test_set, labels, include_date=False):
    scores = model.predict(test_set)
    predictions = tf.argmax(input=scores, axis=1).numpy()

    metrics = {
        'Model evaluation (loss, metrics)': model.evaluate(test_set),
        'Balanced Accuracy': balanced_accuracy_score(labels, predictions),
        'Confusion Matrix': confusion_matrix(labels, predictions).tolist(),
        'ROC AUC Score': roc_auc_score(labels, scores, multi_class="ovr")
    }

    OUTPUT_DIR = RESULTS_ROOT + dir_name
    if include_date:
        OUTPUT_DIR += datetime.now().strftime('%y-%m-%d--%H-%M-%S') + '/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # write metrics to output file
    filename = f'metrics.json'
    with open(OUTPUT_DIR + filename, 'r+') as f:
        data = json.load(f)
        data[note] = metrics
        f.seek(0)               # should reset file position to the beginning.
        json.dump(data, f, indent=4)
        f.truncate()            # remove remaining part

def run_baseline_metrics(clf, X_test, y_test, dir_name, model_name, include_date=False,
                interactive=False):
    y_pred = clf.predict(X_test)
    y_prob_a = clf.predict_proba(X_test)

    metrics = {
        'Score': clf.score(X_test, y_test),
        'Balanced Score': balanced_accuracy_score(y_test, y_pred),
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist(),
        'ROC AUC Score': roc_auc_score(y_test, y_prob_a, multi_class="ovo")
    }

    for k, v in metrics.items():
        logging.info(f'\t{k}: {v}')

    OUTPUT_DIR = RESULTS_ROOT + dir_name
    if include_date:
        OUTPUT_DIR += datetime.now().strftime('%y-%m-%d--%H-%M-%S') + '/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # write metrics to output file
    filename = f'metrics.json'
    with open(OUTPUT_DIR + filename, 'w') as f:
        json.dump(metrics, f, indent=4)

    # plot Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_test, y_pred,
        title=f'{model_name} Confusion Matrix')
    if interactive: plt.show()
    plt.savefig(OUTPUT_DIR + 'confusion_matrix.png')
    
    # plot ROC Curves
    skplt.metrics.plot_roc(y_test, y_prob_a, title=f'{model_name} ROC Curves')
    if interactive: plt.show()
    plt.savefig(OUTPUT_DIR + 'ROC_curves.png')

def get_image_bases(image_root: str) -> list:
    """ Returns a list of sorted image_bases contained in the image_root. The
    image_bases are sorted by the image id, then face id, then augment id.
    (i.e. image-[image id]-[face id]-[augment id].png) """
    return list(sorted(os.listdir(image_root), key=lambda x: tuple(
        int(x.split('.')[0].split('-')[i]) for i in range(1, len(x.split('-'))))))

def get_num_images() -> int:
    return len(os.listdir(IMAGES_ROOT))

def load_dataset(image_root: str):
    """Load dataset from directory where images are split up by class.

    Args:
         image_root: Root folder of the cropped and separated images.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = []
    y = []

    labels = [x for x in os.listdir(image_root) if x[0] != '.']

    for label in labels:
        image_bases = get_image_bases(image_root + label)
        logging.info(f'Loading {len(image_bases)} images ({label})...')
        for image_base in tqdm(image_bases):
            im = np.array(Image.open(image_root + label + '/' + image_base))[:,:,:3]
            im = np.average(im.reshape(im.shape[0]**2, 3), axis=1)
            X.append(im)
            y.append(LABELS[label])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

def build_description(title: str) -> str:
    lines = [
        title,
        '',
        'Charles Pan, Gilbert Rosal, & Dean Stratakos',
        'CS 229: Machine Learning',
        'November 18, 2020'
    ]

    max_length = max(len(l) for l in lines)
    bar = f"+{'-' * (max_length + 2)}+"
    description = f'{bar}\n'
    for line in lines:
        total_padding = max_length - len(line)
        left_padding = int(total_padding / 2)
        right_padding = int((total_padding + 1) / 2)
        description += f"| {' ' * left_padding}{line}{' ' * right_padding} |\n"
    description += f'{bar}'

    return description

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

def get_original_data():
    """ Collects all of the images and annotations from the archive directory.
    Converts the annotation XML files to a list of dictionaries.

    Returns:
        image_bases ([str]): List of the filenames for all images
        annotations ([dict]): List of the parsed annotations
    """
    # sort by the image id (i.e. maksssksksss[image id].png)
    image_bases = list(sorted(os.listdir(IMAGES_ROOT),
                              key=lambda x: int(x[12:-4])))
    annotations = list(sorted(os.listdir(ANNOTATIONS_ROOT),
                              key=lambda x: int(x[12:-4])))
    assert len(image_bases) == len(annotations), \
        f'Number of images ({len(image_bases)}) does not match the number of \
        annotations ({len(annotations)})'
    
    annotations = [parseXML(ANNOTATIONS_ROOT + annotation)
                    for annotation in annotations]

    return image_bases, annotations