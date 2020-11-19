"""
file: run_pipeline.py
---------------------
This script is the entryway for users. All user configurations are exposed
through the command line arguments, which can be seen below.

TODO: Provide an option to run all models at once.
"""
import argparse
from datetime import datetime
import logging

from config import FORMAT
from utils import util

from models import logreg
# from models import resNetPretrained
# from models import resNetUntrained
from models import svm

def parse_args() -> dict:
    """ Parse the input arguments. """
    arg_parser = argparse.ArgumentParser(
        description=util.build_description('Face Mask Detection Pipeline'),
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('model',
        help='specify the model to run', type=str,
        choices=['LogReg', 'SVM', 'ResNetUntrained', 'ResNetPretrained'],
        default='resNetPretrained')
    group = arg_parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', help='increase output verbosity',
        action='store_true')
    group.add_argument('-q', '--quiet', help='decrease output verbosity',
        action='store_true')

    return arg_parser.parse_args()

def main():
    start = datetime.now()
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    args = parse_args()
    if args.model == 'LogReg':
        logreg.main()
    elif args.model == 'SVM':
        svm.main()
    elif args.model == 'ResNetUntrained':
        resNetUntrained.main()
    elif args.model == 'ResNetPretrained':
        resNetPretrained.main()
    
    logging.info('Total runtime: {}'.format(datetime.now() - start))

if __name__ == '__main__':
    main()