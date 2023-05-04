import os
import glob
import time 
import random
import datetime
from easydict import EasyDict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils_general import set_random_seeds
from utils.utils_logging import Logger, ddp_print
from utils.losses import BCELoss
from datasets import get_dataset
from models import UNet

from models import get_model

opt = {
    'EXP_NAME': 'remove',
    'MODEL_NAME': 'UNet',
    'NETWORK_NAME': 'UNet',
    'IN_CHANNELS': 4,
    'NUM_CLASSES': 1,
    'DATASET': 'spacenet6optical',
    'TRAIN_DIR': '/home/yh.sakong/data/preprocessed/optical/train',
    'VAL_DIR': '/home/yh.sakong/data/preprocessed/optical/val',
    'TRAIN_BATCH': 2,
    'VAL_BATCH': 1,
    'NUM_WORKERS': 0,
    'MAX_INTERVAL': 80000,
    'VAL_INTERVAL': 2000,
    'LOG_INTERVAL': 200,
    'BEST_SCORE': 'Dice', # IoU, Dice
    'THRESHOLD': 0.3,
    'CHECKPOINT_DIR': '/home/yh.sakong/github/workspace-segmentation/saved_models',
    'LR': 2e-03,
    'WEIGHT_DECAY': 1e-3,
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
ddp_print('Number of GPUs : ', opt.WORLD_SIZE)

set_random_seeds(random_seed=40)

train_loader, val_loader = get_dataset(opt)
ddp_print('Number of train images: ', len(train_loader.dataset))
ddp_print('Number of val images: ', len(val_loader.dataset))

model = get_model(opt)
# model.load_weights()

optimizer = optim.AdamW(model.net.parameters(), lr=opt.LR)

t = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
opt.SAVE_DIR = os.path.join('./exp', opt.EXP_NAME, t)
os.makedirs(opt.SAVE_DIR, exist_ok=True)
logger = Logger(opt.EXP_NAME, log_path=opt.SAVE_DIR)
ddp_print(f"Log and Checkpoint will be saved '{opt.SAVE_DIR}' \n")

def main():
    
    loss_avg = 0
    dice_avg = 0
    interval = 0
    generator = iter(train_loader) # Iteration based
    while interval < opt.MAX_INTERVAL:
        try:
            interval += 1
            current_lr = model.get_lr(optimizer)
            ###################### Train ######################
            model.train()

            data = next(generator)
            image, label = data['image'].to(rank), data['label'].to(rank)
            output = model.forward(image)

            optimizer.zero_grad()
            loss = model.get_loss(output, label)
            loss.backward()
            optimizer.step()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss_avg += (loss.item() / opt.WORLD_SIZE)

            dice_s = model.get_metric(output, label)
            dice_s_sum = torch.tensor(dice_s, device=rank, dtype=torch.float)
            dist.all_reduce(dice_s_sum, op=dist.ReduceOp.SUM)
            dice_avg += (dice_s_sum.item() / opt.WORLD_SIZE)

            dist.barrier()
            if rank == 0:
                ###################### Logging ######################
                if (((interval % opt.LOG_INTERVAL) == 0) or (interval == opt.MAX_INTERVAL)):

                    msg = (
                        f'[{interval:6d}/{opt.MAX_INTERVAL}] | '
                        f'LR: {current_lr:0.8e} | '
                        f'Loss: {loss_avg/interval:0.4f} | '
                        f'Dice: {dice_avg/interval:0.4f} |'
                    )
                    print(msg)
                    logger.train.info(msg)

            ###################### Validation ######################
            if (((interval % opt.VAL_INTERVAL) == 0) or (interval == opt.MAX_INTERVAL)):
                dist.barrier()

                with torch.no_grad():
                    model.eval()

                    if rank == 0:
                        tbar = tqdm(val_loader, dynamic_ncols=True, desc="Validation")
                    else:
                        tbar = val_loader
                    
                    dice_avg_val = 0
                    dice_avg_val2 = 0
                    for idx, data in enumerate(tbar, start=1):
                        image, label = data['image'].to(rank), data['label'].to(rank)
                        output = model.forward(image)

                        batch = image.shape[0]
                        dice_s = model.get_metric(output, label)
                        dice_s_sum = torch.tensor(dice_s, device=rank, dtype=torch.float)
                        dist.all_reduce(dice_s_sum, op=dist.ReduceOp.SUM)
                        dice_avg_val += ((dice_s_sum.item() / opt.WORLD_SIZE))

                        if rank == 0:
                            ###################### Logging ######################
                            msg = (
                                f'[{interval:6d}/{opt.MAX_INTERVAL}] | '
                                f'Validation | '
                                f'Dice: {dice_avg_val/idx:0.4f} |'
                            )
                            tbar.set_description(msg)
                            if idx == len(val_loader):
                                logger.val.info(msg)                 
                    
                    if rank == 0:
                        score = dice_avg_val/idx
                        model.save_checkpoint(interval, score)

        except StopIteration:
            train_loader.sampler.set_epoch(interval)
            generator = iter(train_loader)


if __name__ == '__main__':
    main()