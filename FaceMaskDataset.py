"""
file: FaceMaskDataset.py
------------------------
Tutorial:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import os
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset

from preprocess import parseXML

class FaceMaskDataset(Dataset):
    def __init__(self, images, root_dir, transform=None):
        self.images = images
        self.images = list(sorted(os.listdir('./archive/images/'), key=lambda x: int(x[12:-4])))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_filename = './archive/images/maksssksksss' + str(idx) + '.png'
        label_filename = './archive/annotations/maksssksksss' + str(idx) + '.xml'        
        image = Image.open(image_filename)
        label = parseXML(label_filename)

        if self.transform:
            image = self.transform(image)

        return image, label