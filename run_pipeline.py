import torch
from torchvision import transforms, datasets, models

import preprocess
from FaceMaskDataset import FaceMaskDataset

def main():
    preprocess.main()

    transform = transforms.Compose([transforms.ToTensor()])
    data = FaceMaskDataset(transform)
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=4,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

if __name__ == '__main__':
    main()