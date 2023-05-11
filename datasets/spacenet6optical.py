import os
import numpy as np
import rasterio as rs

from PIL import Image
from glob import glob

from torch.utils.data import Dataset

class SpaceNet6Optical(Dataset):
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
        try:
            mask = np.array(Image.open(mask_path))
        except:
            mask = rs.open(mask_path).read().transpose(1,2,0)
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