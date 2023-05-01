import os
import glob
import time 
import random
import datetime
from easydict import EasyDict
from decimal import Decimal

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils_general import set_random_seeds
from utils.utils_logging import Logger
from utils.losses import BCELoss
from datasets import get_dataset
from models import UNet

opt = {
    'EXP_NAME': 'beit_adapter_upernet_aux_lr',
    'DATASET': 'spacenet6optical',
    'TRAIN_DIR': '/home/yh.sakong/data/preprocessed/optical/train',
    'VAL_DIR': '/home/yh.sakong/data/preprocessed/optical/val',
    'TRAIN_BATCH': 2,
    'VAL_BATCH': 16,
    'MAX_INTERVAL': 4000,
    'VAL_INTERVAL': 10,
    'LOG_INTERVAL': 1,
    'BEST_SCORE': 'Dice', # IoU, Dice
    'THRESHOLD': 0.3,
    'CHECKPOINT_DIR': '/home/yh.sakong/github/workspace-segmentation/saved_models',
    'NUM_WORKERS': 0,
    'NUM_CLASSES': 8,
    'LR': 2e-05,
    'WEIGHT_DECAY': 1e-3,
    'MODEL_NAME': 'segformer',
    'PRE_TRAINED': False,
    'PRETRAINED_PATH': '/home/yh.sakong/github/distillation/pretrained/beit_large_patch16_224_pt22k_ft22k.pth',
    'DISPLAY_N': 4,
    'RESUME_PATH': '',
    'CHECKPOINT': '/home/yh.sakong/github/new_workspace/saved_models/uentplusplus_deepglobe_newtrain_best.pth'
}
opt = EasyDict(opt)

dist.init_process_group("nccl")
opt.WORLD_SIZE = dist.get_world_size()
rank = dist.get_rank()
torch.cuda.set_device(rank)
print('Number of GPUs : ', opt.WORLD_SIZE)
# device = f'cuda:{rank}'
# set_random_seeds(random_seed=40)

train_loader, val_loader = get_dataset(opt)

model = UNet(4, 1)
model.to(rank)
if opt.WORLD_SIZE > 1:
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[rank], output_device=rank)

optimizer = optim.AdamW(model.parameters(), lr=opt.LR)
loss_fn = BCELoss()

t = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
opt.SAVE_DIR = os.path.join('./exp', opt.EXP_NAME, t)
os.makedirs(opt.SAVE_DIR, exist_ok=True)
logger = Logger(opt.EXP_NAME, log_path=opt.SAVE_DIR)
print(f'Log and Checkpoint will be saved {opt.SAVE_DIR}\n')

def main():
    model.train()
    tbar = tqdm(train_loader, dynamic_ncols=True)

    loss_avg = 0
    interval = 0
    while interval < opt.MAX_INTERVAL:
        for data in tbar:
            image, label = data['image'].to(rank), data['label'].to(rank)
            output = model(image)

            optimizer.zero_grad()
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss_avg += (loss.item() / opt.WORLD_SIZE)
            tbar.set_description(f'[{interval+1}/{opt.MAX_INTERVAL}] | Loss: {loss.item()/opt.WORLD_SIZE}')
            
            interval += 1


    # print(model)
if __name__ == '__main__':
    main()
# opt.WORLD_SIZE = 2
# if __name__ == '__main__':
#     train, val = get_dataset(opt)