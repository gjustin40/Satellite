import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from utils.utils_ddp import reduce_dict, average_gradients 
from .networks import build_network

class BaseModel(ABC):
    def __init__(self, opt, rank):
        self.opt = opt
        self.rank = rank
        self.device = torch.device(f'cuda:{self.rank}')
        
    @abstractmethod
    def set_input(self, data):
        pass
        
    @abstractmethod
    def forward(self):
        pass
        
    @abstractmethod
    def get_loss(self):
        pass

    # def update_lr(self):
    #     pass

    def backward(self):    
        self.optimizer.zero_grad()
        self.get_loss()
        self.loss.backward()
        average_gradients(self.net, world_size=self.opt.WORLD_SIZE)
        self.optimizer.step()
    
    def reduce_loss(self, idx):
        losses_ = {'loss': torch.tensor(self.loss.item()).to(self.device)}
        losses_dict = reduce_dict(losses_, world_size=self.opt.WORLD_SIZE, cpu=True) 
        self.loss_sum += losses_dict['loss'].item()  # loss sum
        self.loss_avg = self.loss_sum / (idx+1)

        return self.loss_avg
    
    def reset(self):
        self.count = 0
        
        self.loss = 0
        self.loss_sum = 0
        self.loss_avg = 0
        
        self.metric_sum = 0
        self.metric_avg = 0
        
    
    # @torch.no_grad()
    # def test(self):
    #     pass
    
    
    # @torch.no_grad()
    # def infer(self):
    #     pass
        
        
    # def metric(self, idx):
    #     metrics_ = {'metric_results': torch.tensor(self.metric_fn(self.label.cpu(), self.output.cpu())).to(self.device)}
    #     metrics_dict = reduce_dict(metrics_, world_size=self.opt.WORLD_SIZE, cpu=True) # Reduce from all gpus
    #     self.metric_sum += np.array(metrics_dict['metric_results'])
    #     self.metric_avg = self.metric_sum / (idx+1)
        
    #     return self.metric_avg
    
    
    # def logging(self, logger, msg, mode='train'):
    #     if mode == 'train':
    #         logger.train.info(msg)
    #     elif mode == 'val':
    #         logger.val.info(msg)
    
    # def create_message(self, epoch, loss_avg, metric_avg, data_size=None, mode=None):
    #     a, b = mode.split('_')
    #     if b == 'description':
    #         msg = f'{a} ({epoch+1}) | Loss: {loss_avg:0.6f} |'
    #         for i, metric in enumerate(['Pre', 'Recall', 'F1', 'IOU']):
    #             msg += f' {metric}: {metric_avg[i]:0.3f} |'
                
    #     elif b == 'log':
    #         msg = f'{a} ({epoch+1})[{self.count}/{data_size}] | Loss: {loss_avg:0.6f} |'
    #         for i, metric in enumerate(['Pre', 'Recall', 'F1', 'IOU']):
    #             msg += f' {metric}: {metric_avg[i]:0.3f} |'
    #     return msg
    
    # def save_checkpoint(self, epoch):
    #     if self.opt.BEST_SCORE == 'f1':
    #         score = self.metric_avg[2]
    #     elif self.opt.BEST_SCORE == 'iou':
    #         score = self.metric_avg[3]
            
    #     if score > self.best:
    #         for param_group in self.optimizer.param_groups:
    #             cur_lr = param_group['lr']
    #         state = {
    #             'state_dict': self.net.module.state_dict(),
    #             'precision': round(self.metric_avg[0], 4),
    #             'recall': round(self.metric_avg[1], 4),
    #             'f1': round(self.metric_avg[2], 4),
    #             'IOU': round(self.metric_avg[3], 4),
    #             'lr': cur_lr,
    #             'epoch': epoch
    #         }
            
    #         if os.path.exists(self.opt.CHECKPOINT_DIR) is False:
    #             os.mkdir(self.opt.CHECKPOINT_DIR)
    #         weight_path = os.path.join(self.opt.CHECKPOINT_DIR, f'{self.opt.EXP_NAME}_best.pth')
    #         torch.save(state, weight_path)
    #         print(f'Saving current weight | Before F1: {self.best} | Current F1: {score} | Current LR: {cur_lr}')
    #         self.best = score

    
    # def visualize(self):
    #     pass
        
    def get_network(self, network_name, num_classes, pretrained=False, resume=False):
        net = build_network(network_name, num_classes, pretrained=pretrained)
        if pretrained:
            print(f'Loading pre-trained of {network_name}....')
        if resume:
            state = torch.load(self.opt.CHECKPOINT, map_location='cpu')
            state_dict = state['state_dict']
            net.load_state_dict(state_dict)
            if resume:
                self.best = state[self.opt.BEST_SCORE]
        net = net.to(self.device)
        if self.opt.WORLD_SIZE > 1:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = DDP(net, device_ids=[self.rank], output_device=self.rank)
        
        return net