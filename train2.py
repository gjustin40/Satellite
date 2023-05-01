import random
import os
import datetime
from easydict import EasyDict
from decimal import Decimal

import glob
import time 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.cuda.amp import GradScaler, autocast
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils_general import set_random_seeds
from utils.utils_logging import Logger
from datasets import get_dataset
from models import UNet


from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import SpaceNet6Optical
from utils.metrics import dice, iou
# from models import FCN

# from utils.utils_ddp import setup_for_distributed
from utils.utils_general import set_random_seeds
from utils.utils_logging import Logger
from utils.losses import BCELoss
# from checkpoint import load_pretrained_weight
# from semseg.models import SegFormer
# from models.beit_adapter_upernet import BEiTAdapterUperNet
# from models.beit_adapter_upernet_aux import BEiTAdapterUperNetAux
# from models import init_weights
# from unet import UNet
# from torchvision.models.segmentation import fcn_resnet101
# import segmentation_models_pytorch as smp

# from semseg import WarmupPolyLR, WarmupLR, PolyLR

# from schedulers import ReduceLROnPlateauPatch

opt = {
    'EXP_NAME': 'beit_adapter_upernet_aux_lr',
    'DATASET': 'spacenet6optical',
    'TRAIN_DIR': '/home/yh.sakong/data/preprocessed/optical/train',
    'VAL_DIR': '/home/yh.sakong/data/preprocessed/optical/val',
    'TRAIN_BATCH': 2,
    'VAL_BATCH': 16,
    'EPOCH': 4000,
    'LOG_INTERVAL': 1,
    'VAL_EPOCH': 10,
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


def train(epoch, model, dataloader, optimizer, world_size, logger, device, scheduler):
    data_size = len(dataloader.dataset)
    local_rank = dist.get_rank()
    
    if local_rank == 0:
        tbar = tqdm(dataloader, dynamic_ncols=True)
    else:
        tbar = dataloader
    model.to(device)
    model.train()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

    loss_fn1 = BCELoss()
    # loss_fn2 = smp.losses.DiceLoss(mode='binary', from_logits=True)

    # scaler = GradScaler()

    total = 0
    epoch_loss = 0
    epoch_dice_s = 0
    epoch_iou_s = 0
    for idx, data in enumerate(tbar):
        image, label = data['image'].to(device), data['label'].to(device) # image: [B, C, H, W], label: [B, H, W]
        total += image.shape[0]
        optimizer.zero_grad()

        # [f1, f2, f3, f4, final] # final: [B, 1, H, W]
        # outputs[-1] : (B, 1, H, W) / label : (B, H, W)

        outputs = model(image)

        # if loss 1개
        loss = loss_fn1(outputs, label)
        # loss2 = loss_fn1(outputs[-2], label)
        # loss = loss1 + loss2 * 0.4
        # if loss 2개
        # loss1 = loss_fn1(outputs[-1], label)
        # loss2 = loss_fn2(outputs[-1], label)
        # loss = loss1 + loss2

        # loss = loss_fn(outputs[-1].squeeze(1), label)        
        loss.backward()
        optimizer.step()
        # scheduler.step()

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss_avg = loss.item() / world_size
        epoch_loss += loss_avg

        # Metric
        # pred = (torch.sigmoid(outputs['out'].cpu()) > opt.THRESHOLD).float()
        pred = (torch.sigmoid(outputs[-1].cpu()) > opt.THRESHOLD).float()

        # remove after test
        dice_remove = torch.tensor(dice(pred, label.cpu()), device=device, dtype=torch.float)
        dist.all_reduce(dice_remove, op=dist.ReduceOp.SUM)

        dice_s = dice(pred, label.cpu())
        dice_s_sum = torch.tensor(dice_s, device=device, dtype=torch.float)
        dist.all_reduce(dice_s_sum, op=dist.ReduceOp.SUM)
        dice_s_avg = dice_s_sum.item() / world_size
        epoch_dice_s += dice_s_avg
        
        iou_s = iou(pred, label.cpu())
        iou_s_sum = torch.tensor(iou_s, device=device, dtype=torch.float)
        dist.all_reduce(iou_s_sum, op=dist.ReduceOp.SUM)
        iou_s_avg = iou_s_sum.item() / world_size
        epoch_iou_s += iou_s_avg

        dist.barrier()

        if local_rank == 0:
            msg = f'Train Epoch[{epoch+1}/{opt.EPOCH}][{idx+1}/{len(dataloader)}] | Loss: {epoch_loss/total:0.4f} | IoU: {epoch_iou_s/(idx+1):0.4f} | Dice: {epoch_dice_s/(idx+1):0.4f} & {dice_s_avg:0.4f} & {dice_remove:0.4f} | LR: {Decimal(scheduler.get_lr()[0]):0.4e}'
            tbar.set_description(msg)
            if ((idx+1) % opt.LOG_INTERVAL == 0) or ((idx + 1) == len(dataloader)):
                logger.train.info(msg)

        
@torch.no_grad()
def val(epoch, model, dataloader, world_size, best, logger, device, scheduler):
    data_size = len(dataloader.dataset)
    local_rank = dist.get_rank()
    
    if local_rank == 0:
        tbar = tqdm(dataloader, dynamic_ncols=True)
    else:
        tbar = dataloader
    model.to(device)
    model.eval()

    loss_fn1 = BCELoss()
    # loss_fn2 = smp.losses.DiceLoss(mode='binary', from_logits=True)

    total = 0
    epoch_loss = 0
    epoch_dice_s = 0
    epoch_iou_s = 0
    for idx, data in enumerate(tbar):
        image, label = data['image'].to(device), data['label'].to(device)
        total += image.shape[0]
        
        outputs = model(image)
        # loss = loss_fn2(outputs['out'], label)
        # loss = loss_fn2(outputs[-1], label)
        loss = loss_fn1(outputs[-1], label)
        # loss2 = loss_fn2(outputs[-1], label)
        # loss = loss1 + loss2
        # loss = loss_fn(outputs[-1].squeeze(1), label)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss_avg = loss.item() / world_size
        epoch_loss += loss_avg

        pred = (torch.sigmoid(outputs[-1].cpu()) > opt.THRESHOLD).float()
        # pred = (torch.sigmoid(outputs['out'].cpu()) > opt.THRESHOLD).float()
        # Dice
        dice_s = dice(pred, label.cpu())
        dice_s_sum = torch.tensor(dice_s, device=device, dtype=torch.float)
        dist.all_reduce(dice_s_sum, op=dist.ReduceOp.SUM)
        dice_s_avg = dice_s_sum.item() / world_size
        epoch_dice_s += dice_s_avg

        # IoU
        iou_s = iou(pred, label.cpu())
        iou_s_sum = torch.tensor(iou_s, device=device, dtype=torch.float)
        dist.all_reduce(iou_s_sum, op=dist.ReduceOp.SUM)
        iou_s_avg = iou_s_sum.item() / world_size
        epoch_iou_s += iou_s_avg

        dist.barrier()

        if local_rank == 0:
            msg = f'Valid Epoch[{epoch+1}/{opt.EPOCH}][{idx+1}/{len(dataloader)}] | Loss: {epoch_loss/total:0.4f} | IoU: {epoch_iou_s/(idx+1):0.4f} | Dice: {epoch_dice_s/(idx+1):0.4f} | LR: {Decimal(scheduler.get_lr()[0]):0.4e}'
            tbar.set_description(msg)
            if ((idx+1) % 5 == 0) or ((idx + 1) == len(dataloader)):
                logger.val.info(msg)

    # Checkpoint
    if local_rank == 0:
        if opt.BEST_SCORE == 'IoU':
            currnet_metric = epoch_iou_s/(idx+1)
        elif opt.BEST_SCORE == 'Dice':
            current_metric = epoch_dice_s/(idx+1)

        if current_metric > best:
            state = {
                'state_dict': model.module.state_dict(),
                'IoU': round(epoch_iou_s/(idx+1), 4),
                'Dice': round(epoch_dice_s/(idx+1), 4),
                'epoch': epoch+1
            }
            pth_list = sorted(glob.glob(opt.SAVE_DIR + '/*.pth'))
            if len(pth_list) > 3:
                os.remove(pth_list[0])
            weight_path = os.path.join(opt.SAVE_DIR, f'best_{epoch+1}_{opt.BEST_SCORE}-{current_metric:0.4f}.pth')
            torch.save(state, weight_path)
            print(f'Saving Current Weight | Before: {opt.BEST_SCORE}-{best:0.4f} | Current: {opt.BEST_SCORE}-{current_metric:0.4f}\n')
            best = current_metric

    return best

    # if local_rank == 0:
    #     msg_log = model.create_message(epoch, loss_avg, metric_avg, data_size=data_size, mode='Val_log')
    #     model.logging(logger, msg_log, mode='val')
    
        # model.save_checkpoint(epoch)



def run(rank, world_size):
    dist.init_process_group("nccl")
    opt.WORLD_SIZE = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    set_random_seeds() #################
    device = f'cuda:{rank}'

    set_random_seeds()


    model = UNet(4, 1)
    model.to(rank)
    if opt.WORLD_SIZE > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], output_device=rank)

    train_loader, val_loader = get_dataset(opt)
    optimizer = optim.AdamW(model.parameters(), lr=opt.LR)

    # max_iter = opt.EPOCH * len(train_loader) + 6000
    # print('Max Iters : ', max_iter)
    # scheduler =  WarmupPolyLR(optimizer, power=1, max_iter=max_iter+3000, warmup_iter=3000, warmup='linear')

    scheduler = ReduceLROnPlateauPatch(optimizer, 'max', patience=10, factor=0.4, threshold=0.0001, threshold_mode='rel')

    if rank == 0:
        t = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
        opt.SAVE_DIR = os.path.join('./exp', opt.EXP_NAME, t)
        os.makedirs(opt.SAVE_DIR, exist_ok=True)
        logger = Logger(opt.EXP_NAME, log_path=opt.SAVE_DIR)
        print(f'Log and Checkpoint will be saved {opt.SAVE_DIR}\n')
    else:
        logger = None

    dist.barrier()
    
    for epoch in range(opt.EPOCH):
        train(epoch, model, train_loader, optimizer, world_size, logger, device, scheduler)
        if (epoch+1) % opt.VAL_EPOCH == 0:
            best = val(epoch, model, val_loader, world_size, best, logger, device, scheduler)
            scheduler.step(best)
        

def init_process(rank, world_size, fn, backend='nccl'):
    # information used for rank 0
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=5400))
    
    fn(rank, world_size)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()