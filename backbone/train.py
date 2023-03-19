import os
import datetime
from easydict import EasyDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision
import torchvision.transforms as transforms

from utils.utils_ddp import setup_for_distributed
from utils.utils_general import set_random_seeds
from utils.utils_logging import Logger
from models import create_model

opt = {
    'MODEL_NAME': 'resnet18',
    'NUM_CLASSES': 10,
    'PRETRAINED': False,
    'RESUME': False,
    'LR': 0.01,
    'EPOCH': 4,
    'VAL_EPOCH': 2
}
opt = EasyDict(opt)

def get_dataset():
    # train_img_dir = '/home/yh.sakong/data/LoveDA/Train_merge/images'
    # train_mask_dir = '/home/yh.sakong/data/LoveDA/Train_merge/masks'
    # val_img_dir = '/home/yh.sakong/data/LoveDA/Val_merge/images'
    # val_mask_dir = '/home/yh.sakong/data/LoveDA/Val_merge/masks'
    

    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    world_size = dist.get_world_size()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_sampler = DistributedSampler(trainset, num_replicas=world_size)
    val_sampler = DistributedSampler(valset, num_replicas=world_size)

    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=False, num_workers=2, sampler=train_sampler, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size = 64, shuffle=False, num_workers=2, sampler=val_sampler, pin_memory=True)

    return train_loader, val_loader


def init_process(rank, world_size, fn, backend='nccl'):
    # information used for rank 0
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=5400))
    setup_for_distributed(rank == 0)
    
    fn(rank, world_size)

def train(epoch, model, dataloader, logger):
    local_rank = dist.get_rank()
    if local_rank == 0:
        tbar = tqdm(dataloader, dynamic_ncols=True)
    else:
        tbar = dataloader

    data_size = len(dataloader.dataset)

    model.reset()
    model.net.train()
    for idx, data in enumerate(tbar):
        model.set_input(data)
        model.forward()
        model.backward()

        # loss_avg = model.reduce_loss(idx)
        # metric_avg = model.metric(idx)
        
        # msg = model.create_message(epoch, loss_avg, metric_avg, mode='Train_description')
        # tbar.set_description(msg)
        # dist.barrier()

    # if local_rank == 0:
    #     msg_log = model.create_message(epoch, loss_avg, metric_avg, data_size=data_size, mode='Val_log')
    #     model.logging(logger, msg_log, mode='val')
    
    #     model.save_checkpoint(epoch)

@torch.no_grad()
def val(epoch, model, dataloader, logger):
    local_rank = dist.get_rank()
    if local_rank == 0:
        tbar = tqdm(dataloader, dynamic_ncols=True)
    else:
        tbar = dataloader

    data_size = len(dataloader.dataset)
    
    model.reset()
    model.net.eval()
    for idx, data in enumerate(tbar):
        model.set_input(data)
        model.forward()
        model.get_loss()
        
        # loss_avg = model.reduce_loss(idx)
        # metric_avg = model.metric(idx)
        # msg = model.create_message(epoch, loss_avg, metric_avg, mode='Val_description')
        # tbar.set_description(msg)
        # dist.barrier()
        
    # if local_rank == 0:
    #     msg_log = model.create_message(epoch, loss_avg, metric_avg, data_size=data_size, mode='Val_log')
    #     model.logging(logger, msg_log, mode='val')
    
    #     model.save_checkpoint(epoch)

def run(rank, world_size):
    torch.cuda.set_device(rank)
    # torch.cuda.empty_cache()
    # device = torch.device(f"cuda:{rank}")
    set_random_seeds()

    opt.WORLD_SIZE = world_size
    model = create_model(opt, rank)
    train_loader, val_loader = get_dataset()
    # logger = Logger(opt.EXP_NAME)
    logger = None
    
    for epoch in range(opt.EPOCH):
        train(epoch, model, train_loader, logger)
        if (epoch+1) % opt.VAL_EPOCH == 0:
            val(epoch, model, val_loader, logger)
        # model.update_lr()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    opt.WORLD_SIZE = world_size

    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):

        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()