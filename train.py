import os
import glob
import time 
import random
import datetime
import argparse
import yaml
from yaml.loader import SafeLoader
from easydict import EasyDict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils_general import set_random_seeds, running_time
from utils.utils_logging import Logger, ddp_print
from datasets import get_dataset

from models import get_model

#################### Read YAML File ####################
parser = argparse.ArgumentParser(description='Train a segmentor')
parser.add_argument('config', help='train config file path')
args = parser.parse_args()
with open(args.config, "r") as f:
    opt = yaml.safe_load(f)

#################### Set DDP ####################
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=18000))
WORLD_SIZE = dist.get_world_size()
RANK = dist.get_rank()
torch.cuda.set_device(RANK)
ddp_print('Number of GPUs : ', WORLD_SIZE)
opt['WORLD_SIZE'] = WORLD_SIZE

#################### Make directory and logger for save Result ####################
t = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
SAVE_DIR = os.path.join('./exp', opt['EXP']['EXP_NAME'], t)
opt['EXP']['SAVE_DIR'] = SAVE_DIR

if RANK == 0:
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = Logger(opt['EXP']['EXP_NAME'], log_path=SAVE_DIR)
    with open(f'{SAVE_DIR}/{opt["EXP"]["EXP_NAME"]}.yaml', 'w') as f:
        yaml.dump(opt, f, sort_keys=False)
        
    print(f"Log and Checkpoint will be saved '{SAVE_DIR}' \n")

#################### Set configs as EasyDict ####################
opt = EasyDict(opt)

#################### Set random seeds ####################
set_random_seeds(random_seed=40)

#################### Get DataLoader ####################
train_loader, val_loader = get_dataset(opt)
ddp_print('Number of train images: ', len(train_loader.dataset))
ddp_print('Number of val images: ', len(val_loader.dataset))

#################### Get Model ####################
model = get_model(opt)

#################### Get Optimizers ####################
optimizer = optim.AdamW(model.net.parameters(), lr=opt.OPTIM.LR)

dist.barrier()
def main():
    timer = running_time(opt.INTERVAL.MAX_INTERVAL)

    loss_avg = 0
    dice_avg = 0
    interval = 0
    generator = iter(train_loader) # Iteration based
    while interval < opt.INTERVAL.MAX_INTERVAL:
        try:
            interval += 1
            current_lr = model.get_lr(optimizer)
            ###################### Train ######################
            model.train()

            data = next(generator)
            image, label = data['image'].to(RANK), data['label'].to(RANK)
            output = model.forward(image)

            optimizer.zero_grad()
            loss = model.get_loss(output, label)
            loss.backward()
            optimizer.step()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss_avg += (loss.item() / opt.WORLD_SIZE)

            dice_s = model.get_metric(output, label).to(RANK)
            dist.all_reduce(dice_s, op=dist.ReduceOp.SUM)
            dice_avg += (dice_s.item() / opt.WORLD_SIZE)

            dist.barrier()
            
            ###################### Logging ######################
            if RANK == 0:
                if (((interval % opt.INTERVAL.LOG_INTERVAL) == 0) or (interval == opt.INTERVAL.MAX_INTERVAL)):
                    timer.end_t = time.time()
                    interval_time, eta = timer.predict(interval)

                    msg = (
                        f'[{interval:6d}/{opt.INTERVAL.MAX_INTERVAL}] | '
                        f'LR: {current_lr:0.8e} | '
                        f'Loss: {loss_avg/interval:0.4f} | '
                        f'Dice: {dice_avg/interval:0.4f} | '
                        f'Time: {interval_time} | '
                        f'ETA: {eta}'
                    )
                    print(msg)
                    logger.train.info(msg)
                    
                    timer.start_t = time.time()

            ###################### Validation ######################
            if (((interval % opt.INTERVAL.VAL_INTERVAL) == 0) or (interval == opt.INTERVAL.MAX_INTERVAL)):
                dist.barrier()

                with torch.no_grad():
                    model.eval()

                    if RANK == 0:
                        tbar = tqdm(val_loader, dynamic_ncols=True, desc="Validation")
                    else:
                        tbar = val_loader
                    
                    dice_avg_val = 0
                    for idx, data in enumerate(tbar, start=1):
                        image, label = data['image'].to(RANK), data['label'].to(RANK)
                        output = model.forward(image)

                        dice_s = model.get_metric(output, label).to(RANK)
                        dist.all_reduce(dice_s, op=dist.ReduceOp.SUM)
                        dice_avg_val += ((dice_s.item() / opt.WORLD_SIZE))

                        if RANK == 0:
                            ###################### Logging ######################
                            msg = (
                                f'[{interval:6d}/{opt.INTERVAL.MAX_INTERVAL}] | '
                                f'Validation | '
                                f'Dice: {dice_avg_val/idx:0.4f} |'
                            )
                            tbar.set_description(msg)
                            if idx == len(val_loader):
                                logger.val.info(msg)                 
                    
                    if RANK == 0:
                        state = {
                            'interval': interval,
                            'Dice': round(score, 4),
                            'state_dict': model.net.module.state_dict(),
                        }
                        score = dice_avg_val/idx
                        model.save_checkpoint(interval, score)

                timer.start_t = time.time()

        except StopIteration:
            train_loader.sampler.set_epoch(interval)
            generator = iter(train_loader)


if __name__ == '__main__':
    main()