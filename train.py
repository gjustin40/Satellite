import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
import glob
import time 
import random
import datetime
import argparse
import yaml
import shutil
from yaml.loader import SafeLoader
from easydict import EasyDict

import numpy as np
from tqdm import tqdm

import wandb
from wandb import AlertLevel

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils_general import set_random_seeds, running_time
from utils.utils_logging import Logger, ddp_print
from utils.metrics import MetricTracker
from datasets import get_dataset

from models import get_model

from utils import WarmupPolyLR


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

#################### Set configs as EasyDict ####################
# EasyDict을 여기서 사용한 이유는 yaml을 저장할 easydict으로 저장하면 이상한 것들도 같이 저장돼서...
opt = EasyDict(opt)
if opt.MODEL.IS_RESUME:
    assert opt.MODEL.PRETRAINED_PATH, 'we need PRETRINED_PATH for resume training.'
    ddp_print('Resume Training....')

#################### Set WandDB ####################
if RANK == 0:
    wandb.login()
    wandb.init(
        project="SpaceNet6-Distillation",
        config=opt,
        name=f'{opt.EXP.EXP_NAME}_{t}',
        dir=SAVE_DIR
        )
dist.barrier()

#################### Set random seeds ####################
set_random_seeds(random_seed=40)

#################### Get DataLoader ####################
train_loader, val_loader = get_dataset(opt)
ddp_print('Number of train images: ', len(train_loader.dataset))
ddp_print('Number of val images: ', len(val_loader.dataset))

#################### Get Model ####################
model = get_model(opt)
if RANK == 0:
    # save model.py
    shutil.copyfile(
        f'./models/{opt.MODEL.MODEL_NAME}_model.py',
        f'{SAVE_DIR}/{opt.MODEL.MODEL_NAME}_model.py'
    )    
    print(f"Log and Checkpoint will be saved '{SAVE_DIR}' \n")
    wandb.watch(model.net.module) # # Wandb Logging

#################### Get Optimizers ####################
params = model.get_params()
optimizer = optim.AdamW(params, lr=opt.OPTIM.LR, weight_decay=opt.OPTIM.WEIGHT_DECAY)
scheduler = WarmupPolyLR(optimizer, power=1, max_iter=opt.INTERVAL.MAX_INTERVAL, warmup_iter=1500, warmup='linear')
interval = 0
if opt.MODEL.IS_RESUME:
    ddp_print('Ready for Resume....')
    checkpoint = torch.load(opt.MODEL.PRETRAINED_PATH, map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    interval = checkpoint['interval']
    model.best = checkpoint['metrics'][opt.CHECKPOINT.BEST_METRIC]
    
dist.barrier()
def main():
    timer = running_time(opt.INTERVAL.MAX_INTERVAL)

    train_metric = MetricTracker(opt)
    
    interval = 0
    loss_avg = 0
    generator = iter(train_loader) # Iteration based

    ddp_print('Start Loop....')
    while interval < opt.INTERVAL.MAX_INTERVAL:
        try:
            interval += 1
            current_lr = model.get_lr(optimizer)
            ###################### Train ######################
            model.train()

            data = next(generator)
            if isinstance(data['image'], list):
                image = [img.to(RANK) for img in data['image']]
                # label = [lab.to(RANK) for lab in data['label']]
                label = data['label'].to(RANK)
            else:
                image, label = data['image'].to(RANK), data['label'].to(RANK)

            # remove
            output = model.forward(image)

            optimizer.zero_grad()
            loss = model.get_loss(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss_avg += (loss.item() / opt.WORLD_SIZE)

            # train_avg = train_metric.get(output, label.cpu(), RANK)
            train_avg = train_metric.get(output, label, RANK)

            dist.barrier()
            
            ###################### Logging ######################
            if RANK == 0:

                # Wandb Logging
                w_log = {
                    'LR': current_lr,
                    'Train_loss': loss_avg/interval,
                }
                w_log.update([(f'Train_{m}', s) for m,s in train_avg.items()])
                wandb.log(w_log)

                if (((interval % opt.INTERVAL.LOG_INTERVAL) == 0) or (interval == opt.INTERVAL.MAX_INTERVAL)):
                    timer.end_t = time.time()
                    interval_time, eta = timer.predict(interval)

                    msg = (
                        f'[{interval:6d}/{opt.INTERVAL.MAX_INTERVAL}] | '
                        f'LR: {current_lr:.8e} | '
                        f'Loss: {loss_avg/interval:.4f} | '
                        # f'{" | ".join([f"{m}: {s:.4f}" for m, s in train_avg.items()])} | '
                        f'{" | ".join([f"{m}: {s:.4f}" for m, s in train_avg.items()])} | '
                        f'Time: {interval_time} | '
                        f'ETA: {eta} | '
                        f'{time.strftime("%m%d %H:%M", time.localtime(time.time()))}'
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
                    
                    val_metric = MetricTracker(opt)

                    for idx, data in enumerate(tbar, start=1):
                        if isinstance(data['image'], list):
                            image = [img.to(RANK) for img in data['image']]
                            # label = [lab.to(RANK) for lab in data['label']]
                            label = data['label'].to(RANK)
                        else:
                            image, label = data['image'].to(RANK), data['label']

                        output = model.forward(image)

                        val_avg = val_metric.get(output, label, RANK)
                        if RANK == 0:
                            ###################### Logging ######################
                            msg = (
                                f'[{interval:6d}/{opt.INTERVAL.MAX_INTERVAL}] | '
                                f'Validation | '
                                f'{" | ".join([f"{m}: {s:.4f}" for m, s in val_avg.items()])} | '
                            )
                            tbar.set_description(msg)
                            if idx == len(val_loader):
                                logger.val.info(msg)                 
                    
                    if RANK == 0:
                        # Wandb Logging
                        wandb.log({f'Val_{m}':s for m,s in val_avg.items()})
                        if model.best < val_avg[opt.CHECKPOINT.BEST_METRIC]:
                            alert = (
                                f"Metric: {opt.CHECKPOINT.BEST_METRIC} | "
                                f"{model.best:0.4f} -> {val_avg[opt.CHECKPOINT.BEST_METRIC]:0.4f}"
                                )
                            print('alert')
                            wandb.alert(
                                title=f'Metric Update {interval}',
                                text=alert,
                                level=AlertLevel.INFO)

                        # Checkpoint
                        state = {
                            'interval': interval,
                            'state_dict': model.net.module.state_dict() if opt.WORLD_SIZE > 1 else model.net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                            'metrics': val_avg
                        }
                        model.save_checkpoint(state)

                        

                timer.start_t = time.time()

        except StopIteration:
            train_loader.sampler.set_epoch(interval)
            generator = iter(train_loader)
            train_metric = MetricTracker(opt) # init train_metric


if __name__ == '__main__':
    main()