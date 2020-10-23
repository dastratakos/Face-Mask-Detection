"""
file: FaceMaskDataset.py
------------------------
Tutorial:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
# import os

# from PIL import Image
# from skimage import io, transform
# from torch.utils.data import Dataset

# from config import IMAGE_ROOT, ANNOTATION_ROOT
# from data_processing.preprocess import parseXML
# from utils import util

# class FaceMaskDataset(Dataset):
#     def __init__(self, images, root_dir, transform=None):
#         self.images = images
#         self.images = util.get_image_bases(IMAGE_ROOT)
#         self.transform = transform

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image_filename = IMAGE_ROOT + f'maksssksksss{idx}.png'
#         label_filename = ANNOTATION_ROOT + f'maksssksksss{idx}.xml'        
#         image = Image.open(image_filename)
#         label = parseXML(label_filename)

#         if self.transform:
#             image = self.transform(image)

#         return image, label