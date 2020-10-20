import argparse
from argparse import RawTextHelpFormatter
# import torch
# from torchvision import transforms, datasets, models

import preprocess
# from FaceMaskDataset import FaceMaskDataset

def parse_args():
    """ Parse the input arguments """
    arg_parser = argparse.ArgumentParser(
        description="+------------------------------+\n| Face Mask Detection Pipeline |\n+------------------------------+",
        formatter_class=RawTextHelpFormatter)
    arg_parser.add_argument("model", help="1 for logistic regression, 2 for SVM, 3 for CNN",
        type=int, choices=[1, 2, 3], default=3)
    group = arg_parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    group.add_argument("-q", "--quiet", help="decrease output verbosity", action="store_true")

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

def main():
    args = parse_args()

    preprocess.main()

    # transform = transforms.Compose([transforms.ToTensor()])
    # data = FaceMaskDataset(transform)
    # data_loader = torch.utils.data.DataLoader(
    #     data,
    #     batch_size=4,
    #     collate_fn=lambda batch: tuple(zip(*batch))
    # )

if __name__ == '__main__':
    main()