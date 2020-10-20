"""
file: preprocess.py
-------------------
Parses XML files.
"""
import os
import pprint
import xml.etree.ElementTree as ET

from config import ARCHIVE_ROOT, IMAGE_ROOT, ANNOTATION_ROOT

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

def single_file_test():
    """ Tests the parseXML function for a single XML file.
    """
    pp = pprint.PrettyPrinter(indent=4)

    annotation = parseXML(f'{ANNOTATION_ROOT}maksssksksss0.xml')
    pp.pprint(annotation)

def main():
    """ Collects all of the images and annotations from the archive directory.
    Converts the annotation XML files to a list of dictionaries.

    Returns:
        image_bases ([str]): List of the filenames for all images
        annotations ([dict]): List of the parsed annotations
    """

    # sort by the image id (i.e. maksssksksss[id].png)
    image_bases = list(sorted(os.listdir(IMAGE_ROOT),
                              key=lambda x: int(x[12:-4])))
    annotations = list(sorted(os.listdir(ANNOTATION_ROOT),
                              key=lambda x: int(x[12:-4])))
    assert len(image_bases) == len(annotations), \
        f'Number of images ({len(image_bases)}) does not match the number of \
        annotations ({len(annotations)})'
    
    print(f'The dataset contains {len(image_bases)} data points')
    annotations = [parseXML(f'{ANNOTATION_ROOT}{annotation}')
                    for annotation in annotations]

    return image_bases, annotations

if __name__ == '__main__':
    main()
    # single_file_test()
    