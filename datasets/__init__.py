from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import cv2
from PIL import Image
from .spacenet6optical import SpaceNet6Optical
from .spacenet6sar import SpaceNet6SAR
from .spacenet6kd import SpaceNet6KD

def get_dataset(opt):

    datasets_dict = {
        'spacenet6optical': SpaceNet6Optical,
        'spacenet6sar': SpaceNet6SAR,
        'spacenet6kd': SpaceNet6KD
    }
    
    # train_dir = '/home/yh.sakong/data/sn6_building/preprocessed/rgb_png/train'
    # val_dir = '/home/yh.sakong/data/sn6_building/preprocessed/rgb_png/val'

    train_dir = opt.DATA.TRAIN_DIR
    val_dir = opt.DATA.VAL_DIR

    transform_train = A.Compose([
        # A.Rotate(limit=40,p=0.5),
        A.Resize(512, 512, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.ColorJitter(p=0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
        # A.Normalize(mean=(123.675, 116.28, 103.53, 103.53), std=(58.395, 57.12, 57.375, 57.375)),
        NewNorm(mean=(123.675, 116.28, 103.53, 103.53), std=(58.395, 57.12, 57.375, 57.375)),
        ToTensorV2()])
    
    transform_val = A.Compose([
        A.Resize(512, 512, always_apply=True),
        # A.Normalize(mean=(0.485, 0.456, 0.406, 0.400), std=(0.229, 0.224, 0.225, 0.226)),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # A.Normalize(mean=(123.675, 116.28, 103.53, 103.53), std=(58.395, 57.12, 57.375, 57.375)),
        NewNorm(mean=(123.675, 116.28, 103.53, 103.53), std=(58.395, 57.12, 57.375, 57.375)),
        ToTensorV2()])

    # if opt.TEST.TEST_PATH:
    #     print('Test Mode running....')
    #     testset = datasets_dict[opt.TEST.DATASET](opt.TEST.TEST_DIR, transform_val)
    #     test_loader = DataLoader(testset, batch_size=opt.TEST.TEST_BATCH, shuffle=False, num_workers=opt.TEST.NUM_WORKERS)
    #     return test_loader

    trainset = datasets_dict[opt.DATA.DATASET](train_dir, transform_train)
    valset = datasets_dict[opt.DATA.DATASET](val_dir, transform_val)
    train_sampler = DistributedSampler(trainset, num_replicas=opt.WORLD_SIZE, drop_last=False)
    val_sampler = DistributedSampler(valset, num_replicas=opt.WORLD_SIZE, drop_last=False)
    train_loader = DataLoader(trainset, batch_size=opt.DATA.TRAIN_BATCH, shuffle=False, num_workers=opt.DATA.NUM_WORKERS, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=opt.DATA.VAL_BATCH, shuffle=False, num_workers=opt.DATA.NUM_WORKERS, sampler=val_sampler, pin_memory=True)
    
    
    return train_loader, val_loader


def new_normalize(img, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    img = img.copy().astype(np.float32)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))

    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


class NewNorm(ImageOnlyTransform):
    def __init__(
        self, 
        mean=(123.675, 116.28, 103.53, 103.53),
        std=(58.395, 57.12, 57.375, 57.375),
        always_apply=True, 
        p=1,):
        super(NewNorm, self).__init__(always_apply, p)

        self.mean = mean
        self.std = std

    def apply(self, img, **params):
        return new_normalize(img, self.mean, self.std)