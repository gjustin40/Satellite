import os
from abc import ABC, abstractmethod

# import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# import numpy as np

# from utils.utils_ddp import reduce_dict, average_gradients 
from .networks import get_network

class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.rank = dist.get_rank()


    @abstractmethod
    def forward(self, image):
        # self.ouput = self.net(self.image)
        pass


    # @abstractmethod
    def get_loss(self):
        # self.loss = loss
        pass


    def train(self):
        self.net.train()


    def eval(self):
        self.net.eval()


    def _get_network(self):
        net = get_network(
            network_name=self.opt.NETWORK_NAME,
            in_channels=self.opt.IN_CHANNELS,
            num_classes=self.opt.NUM_CLASSES
        )
        # net.apply(self._init_weights)

        return self._load_checkpoint(net)
    

    def _load_checkpoint(self, net):
        net.to(self.rank)
        if self.opt.WORLD_SIZE > 1:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net, device_ids=[self.rank], output_device=self.rank)

        return net


    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']