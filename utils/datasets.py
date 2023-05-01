import random
import os
from PIL import Image
import torch 
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from glob import glob
import numpy as np

import rasterio as rs

class SpaceNet6Dataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'labels')

        self.image_ids = self.get_ids_(self.image_dir)
        self.mask_ids = self.get_ids_(self.mask_dir)

        assert self.image_ids == self.mask_ids

    def get_ids_(self, dir):
        path_list = glob(os.path.join(dir, '*'))
        ids = list(map(lambda x: os.path.basename(x), path_list))
        return ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        basename = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, basename)
        mask_path = os.path.join(self.mask_dir, basename)

        try:
            image = np.array(Image.open(image_path))
        except:
            image = rs.open(image_path).read().transpose(1,2,0)
        mask = np.array(Image.open(mask_path))
        mask = np.where(mask >= 1, 1, 0)
        if mask.ndim == 3:
            mask = mask[:,:,0]

        augmentations = self.transform(image=image, mask=mask)
        image = augmentations['image'].float()
        mask = augmentations['mask'].long()

        data = {
            'image': image.float(),
            'label': mask.float(),
            'image_path': image_path
        }

        return data

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import numpy as np

    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    def set_random_seeds(random_seed=220678853):

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    set_random_seeds()

    transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(p=0.5),
        # A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
        ToTensorV2()])

    train_dir = '/home/yh.sakong/data/massachusetts_building/png_512/train'
    val_dir = '/home/yh.sakong/data/massachusetts_building/png_512/val'

    trainset = SpaceNet6Dataset(train_dir, transform_train)

    valset = SpaceNet6Dataset(val_dir, transform_train)
    
    trainloader = DataLoader(trainset, batch_size=2, shuffle=False, num_workers=1)
    for data in trainloader:
        print(data['image'].shape, data['label'].shape)
        print(data['image'])
        break
