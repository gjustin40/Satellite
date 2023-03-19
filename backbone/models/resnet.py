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

        # Metric Function

        # Optimizer
        self.optimizer = optim.SGD(self.net.parameters(), lr = self.opt.LR)

    def set_input(self, data):
        self.image = data[0].to(self.device)
        self.label = data[1].to(self.device)

        # Sum all of batch from each GPU
        c = {'count': torch.tensor(self.image.shape[0]).to(self.device)}
        count_dict = reduce_dict(c, world_size=self.opt.WORLD_SIZE, average=False, cpu=True)
        # self.count += count_dict['count'].item()
    def forward(self):
        self.output = self.net(self.image)

    def get_loss(self):
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = self.loss_fn(self.output, self.label)