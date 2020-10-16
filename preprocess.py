"""
file: preprocess.py
-------------------
Parses XML files and formats images.

See below for more information on loading images using PyTorch
https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2

NOTE: another option for parsing XML files
from bs4 import BeautifulSoup
NOTE: another way to crawl through a directory
for root, dirnames, filenames in os.walk("/foo/bar"):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
"""
import os
import pprint
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import helper

# from scipy import ndimage, misc

def parseXML(xml_filename):
    """ This function generates an annotation dictionary representation of
    the contents of the specified XML filename.

    Args:
        xml_filename (String): Relative path to the XML file

    Returns:
        Dictionary: Dictionary representation of the entire XML file
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

def format_images(data_dir):
    # apply transforms
    train_transforms = transforms.Compose([
                                    transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])
    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)                                       

    # load data
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
    return trainloader

def single_file_test():
    """ Tests the parseXML function for a single XML file.
    """
    pp = pprint.PrettyPrinter(indent=4)

    annotation = parseXML('./archive/annotations/maksssksksss0.xml')
    print('=' * 30)
    pp.pprint(annotation)

    image = format_image('./archive/images/maksssksksss0.png')
    print(image)

def main():
    """ Collects all of the images and annotations from the ./archive/ directory.
    Converts the annotation XML files to a list of dictionaries.
    """
    
    imgs = list(sorted(os.listdir('./archive/images/'), key=lambda x: int(x[12:-4])))
    annotations = list(sorted(os.listdir('./archive/annotations/'), key=lambda x: int(x[12:-4])))
    assert len(imgs) == len(annotations), f'Number of images ({len(imgs)}) does not match the number of annotations ({len(annotations)})'
    
    print(f'The dataset contains {len(imgs)} data points')
    annotations = [parseXML('./archive/annotations/' + annotation) for annotation in annotations]

if __name__ == '__main__':
    # main()
    single_file_test()
    