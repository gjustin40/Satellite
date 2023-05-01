from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .spacenet6optical import SpaceNet6Optical

def get_dataset(opt):

    datasets_dict = {
        'spacenet6optical': SpaceNet6Optical
    }
    
    # train_dir = '/home/yh.sakong/data/sn6_building/preprocessed/rgb_png/train'
    # val_dir = '/home/yh.sakong/data/sn6_building/preprocessed/rgb_png/val'

    train_dir = opt.TRAIN_DIR
    val_dir = opt.VAL_DIR

    transform_train = A.Compose([
        # A.Rotate(limit=40,p=0.5),
        A.Resize(512, 512, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.ColorJitter(p=0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
        # A.Normalize(mean=(0.485, 0.456, 0.406, 0.400), std=(0.229, 0.224, 0.225, 0.226)),
        ToTensorV2()])
    
    transform_val = A.Compose([
        A.Resize(512, 512, always_apply=True),
        # A.Normalize(mean=(0.485, 0.456, 0.406, 0.400), std=(0.229, 0.224, 0.225, 0.226)),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
        ToTensorV2()])


    trainset = datasets_dict[opt.DATASET](train_dir, transform_train)
    valset = datasets_dict[opt.DATASET](val_dir, transform_val)
    train_sampler = DistributedSampler(trainset, num_replicas=opt.WORLD_SIZE, drop_last=False)
    val_sampler = DistributedSampler(valset, num_replicas=opt.WORLD_SIZE, drop_last=False)
    train_loader = DataLoader(trainset, batch_size=opt.TRAIN_BATCH, shuffle=False, num_workers=opt.NUM_WORKERS, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=opt.VAL_BATCH, shuffle=False, num_workers=opt.NUM_WORKERS, sampler=val_sampler, pin_memory=True)
    
    return train_loader, val_loader