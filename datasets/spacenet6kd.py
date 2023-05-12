import os
import numpy as np
import rasterio as rs

from PIL import Image
from glob import glob

from torch.utils.data import Dataset

class SpaceNet6KD(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.dir_optical = os.path.join(self.root, 'optical')
        self.dir_sar = os.path.join(self.root, 'sar')

        self.dir_optical_image = os.path.join(self.dir_optical, 'images')
        self.dir_optical_mask = os.path.join(self.dir_optical, 'labels')
        self.dir_sar_image = os.path.join(self.dir_sar, 'images')
        self.dir_sar_mask = os.path.join(self.dir_sar, 'labels')

        self.transform = transform

        self.ids_optical_image = self.get_ids_(self.dir_optical_image)
        self.ids_optical_mask = self.get_ids_(self.dir_optical_mask)
        self.ids_sar_image = self.get_ids_(self.dir_sar_image)
        self.ids_sar_mask = self.get_ids_(self.dir_sar_mask)

        assert self.ids_optical_image == self.ids_optical_mask
        assert self.ids_sar_image == self.ids_sar_mask

    def get_ids_(self, dir):
        path_list = glob(os.path.join(dir, '*'))
        ids = list(map(lambda x: os.path.basename(x), path_list))
        return ids

    def __len__(self):
        return len(self.ids_optical_image)

    def __getitem__(self, idx):
        basename_optical = self.ids_optical_image[idx]
        basename_sar = basename_optical.replace('PS-RGB', 'SAR-Intensity')

        optical_image_path = os.path.join(self.dir_optical_image, basename_optical)
        optical_mask_path = os.path.join(self.dir_optical_mask, basename_optical)
        sar_image_path = os.path.join(self.dir_sar_image, basename_sar)
        sar_mask_path =  os.path.join(self.dir_sar_mask, basename_sar)

        optical_image = rs.open(optical_image_path).read().transpose(1,2,0)
        optical_mask = rs.open(optical_mask_path).read().transpose(1,2,0)
        sar_image = rs.open(sar_image_path).read().transpose(1,2,0)
        sar_mask = rs.open(sar_mask_path).read().transpose(1,2,0)

        optical_mask = np.where(optical_mask >= 1, 1, 0)
        sar_mask  = np.where(sar_mask >= 1, 1, 0)
        if optical_mask.ndim == 3:
            optical_mask = optical_mask[:,:,0]
        if sar_mask.ndim == 3:
            sar_mask = sar_mask[:,:,0]
        
        aug_optical = self.transform(image=optical_image, mask=optical_mask)
        aug_sar = self.transform(image=sar_image, mask=sar_mask)

        image_list = [aug_optical['image'].float(), aug_sar['image'].float()]
        mask       = aug_sar['mask'].float()
        path_list  = [optical_image_path, sar_image_path]

        data = {
            'image': image_list,
            'label': mask,
            'image_path': path_list
        }

        return data