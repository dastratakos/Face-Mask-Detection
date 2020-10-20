import argparse
from argparse import RawTextHelpFormatter

import preprocess

def build_description() -> str:
    lines = [
        'Face Mask Detection Pipeline',
        '',
        'Charles Pan, Gilbert Rosal, & Dean Stratakos',
        'CS 229: Machine Learning',
        'October 20, 2020'
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

def parse_args() -> dict:
    """ Parse the input arguments """
    arg_parser = argparse.ArgumentParser(description=build_description(),
        formatter_class=RawTextHelpFormatter)
    arg_parser.add_argument("model",
        help="1 for logistic regression, 2 for SVM, 3 for CNN", type=int,
        choices=[1, 2, 3], default=3)
    group = arg_parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", help="increase output verbosity",
        action="store_true")
    group.add_argument("-q", "--quiet", help="decrease output verbosity",
        action="store_true")

    args = arg_parser.parse_args()

    if args.model == 1:
        print("Running logistic regression")
    elif args.model == 2:
        print("Running Support Vector Machine")
    elif args.model == 3:
        print("Running Convolutional Neural Network")
    if args.verbose:
        print("Printing more output", "~" * 10)
    elif args.quiet:
        print("Printing less output")
    else:
        print("Printing regular amount of output")
    return args

def main():
    args = parse_args()
    preprocess.main()

if __name__ == '__main__':
    main()