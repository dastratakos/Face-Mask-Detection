"""
file: run_pipeline.py
---------------------
This script is the entryway for users. All user configurations are exposed
through the command line arguments, which can be seen below.
"""
import argparse
from datetime import datetime
import logging
import os

from config import BALANCED_ROOT, CROPPED_ROOT, IMAGES_ROOT, ANNOTATIONS_ROOT, FORMAT
from utils import crop, balance, visualize, util
from models import logreg, svm, resNetPretrained, resNetUntrained

def parse_args() -> dict:
    """ Parse the input arguments. """
    arg_parser = argparse.ArgumentParser(
        description=util.build_description('Face Mask Detection Pipeline'),
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-m', '--model',
        help='specify the model to run', type=str,
        choices=['LogReg', 'SVM', 'ResNetUntrained', 'ResNetPretrained'],
        default='None')
    arg_parser.add_argument('-v', '--visualize', action='store_true',
        help='create a directory of the original images with bounding boxes')

    return arg_parser.parse_args()

def check_original_dataset():
    assert os.path.isdir(IMAGES_ROOT) and os.path.isdir(ANNOTATIONS_ROOT), \
        'Please download the dataset from Kaggle: ' + \
        'https://www.kaggle.com/andrewmvd/face-mask-detection'
    return True

def main():
    start = datetime.now()
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    args = parse_args()

    if args.visualize and check_original_dataset():
        visualize.main()
    if not os.path.isdir(CROPPED_ROOT) and check_original_dataset():
        crop.main()
    if not os.path.isdir(BALANCED_ROOT):
        balance.main()

    if args.model == 'LogReg':
        logreg.main()
    elif args.model == 'SVM':
        svm.main()
    elif args.model == 'ResNetUntrained':
        resNetUntrained.main()
    elif args.model == 'ResNetPretrained':
        resNetPretrained.main()
    elif args.model == 'None':
        print('No model specified!')
    
    logging.info(f'Total runtime: {datetime.now() - start}\n')

if __name__ == '__main__':
    main()