import torch
import torch.nn as nn
import torch.optim as optim

from utils.losses import BCEFocalLoss
from utils.utils_ddp import reduce_dict 
from utils.metrics import BinaryMetrics

from .base import BaseModel

class ResNet(BaseModel):
    def __init__(self, opt, rank):
        super(BaseModel, self).__init__()

        self.opt = opt
        self.rank = rank
        self.device = torch.device(f'cuda:{self.rank}')

        # Define Network
        self.net = self.get_network(
            self.opt.MODEL_NAME,
            self.opt.NUM_CLASSES, 
            self.opt.PRETRAINED,
            self.opt.RESUME
        )

        # Optimizer
        self.optimizer = optim.SGD(self.net.parameters(), lr = self.opt.LR)

    def set_input(self, data):
        self.image = data[0].to(self.device) # shape : (batch, Channel, H, W)
        self.label = data[1].to(self.device) # shape : (batch)
        

    def forward(self):
        self.output = self.net(self.image) # probability, shape : (batch, num_classes)


    def get_loss(self):
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = self.loss_fn(self.output, self.label) # shape : (1)