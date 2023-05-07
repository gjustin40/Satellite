import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .networks import get_network
from utils import load_pretrained_weight


class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.rank = dist.get_rank()

        self.resume_interval = None
        
        self.best = 0
        self.best_checkpoint_path = None

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


    def get_params(self):
        """
        만약 ViT, BEiT 모델을 사용한다면
        layer_decay_optimizer_constructor를 사용 
        """
        return self.net.parameters()


    def _get_network(self):
        net = get_network(
            network_name=self.opt.MODEL.NETWORK_NAME,
            in_chans=self.opt.MODEL.IN_CHANNELS,
            num_classes=self.opt.MODEL.NUM_CLASSES
        )
        # net.apply(self._init_weights)
        print('aaa')
        if self.opt.MODEL.LOAD_PATH:
            checkpoint = torch.load(self.opt.MODEL.LOAD_PATH, map_location='cpu')
            net.load_state_dict(checkpoint['state_dict'])
            if self.rank == 0:
                print('Loading checkpoint.....')

        # if self.opt.MODEL.PRETRAINED_PATH:
        #     net = load_pretrained_weight(net, self.opt.MODEL.PRETRAINED_PATH)
        
        return self._wrap_ddp(net)
    

    def _wrap_ddp(self, net):
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


    def save_checkpoint(self, state):
        # Last Checkpoint Save
        checkpoint_path = os.path.join(self.opt.EXP.SAVE_DIR, 'last.pth')
        torch.save(state, checkpoint_path)

        # Best Score Save
        best_score = state['metrics'][self.opt.CHECKPOINT.BEST_METRIC]
        if self.best < best_score:
            checkpoint_path = os.path.join(
                self.opt.EXP.SAVE_DIR,
                f'best_{state["interval"]}_{self.opt.CHECKPOINT.BEST_METRIC}_{best_score:0.4f}.pth'
            )
            torch.save(state, checkpoint_path)
            if self.best_checkpoint_path is not None:
                os.remove(self.best_checkpoint_path)

            print(
                f"Save Checkpoint '{self.opt.EXP.SAVE_DIR}' | "
                f"Metric : {self.opt.CHECKPOINT.BEST_METRIC} | "
                f"{self.best:0.4f} -> {best_score:0.4f}\n"
            )

            self.best_checkpoint_path = checkpoint_path
            self.best = best_score