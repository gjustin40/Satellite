from utils import layer_decay_optimizer_constructor
import yaml
from easydict import EasyDict
from models import get_model
import torch.distributed as dist
import torch

with open('configs/config.yaml', "r") as f:
    opt = yaml.safe_load(f)

dist.init_process_group("nccl")
WORLD_SIZE = dist.get_world_size()
RANK = dist.get_rank()
torch.cuda.set_device(RANK)

opt['WORLD_SIZE'] = WORLD_SIZE


opt = EasyDict(opt)
model = get_model(opt)

print(layer_decay_optimizer_constructor(opt, model.net))