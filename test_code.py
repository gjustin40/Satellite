# from utils import layer_decay_optimizer_constructor
# import yaml
# from easydict import EasyDict
# from models import get_model
# import torch.distributed as dist
# import torch
# from datasets import get_dataset

# with open('configs/config_beit_adapter_upernet_kd.yaml', "r") as f:
#     opt = yaml.safe_load(f)

# dist.init_process_group("nccl")

# WORLD_SIZE = dist.get_world_size()
# RANK = dist.get_rank()
# torch.cuda.set_device(RANK)

# opt['WORLD_SIZE'] = WORLD_SIZE


# opt = EasyDict(opt)
# model = get_model(opt)

# print(layer_decay_optimizer_constructor(opt, model.net))
############### Test Model #####################
# from models import BEiTAdapterUperNetKD
from models import BEiTAdapterUperNetKD
import torch

net = BEiTAdapterUperNetKD(4, 1).to('cuda:0')
inp = torch.Tensor(2,4,512,512).to('cuda:0')
output = net(inp)
import time
time.sleep(10)
names = [
    'featuers_opt',
    'features_sar',
    'decode_opt',
    'decode_sar',
    'out_opt',
    'out_sar',
    'out_combine'
]
for idx, o in enumerate(output):
    print(names[idx])
    if isinstance(o, list):
        for o1 in o:
            print(o1.shape)
    else:
        print(o.shape)

############### Test Dataset ###################
# opt = EasyDict(opt)
# train_loader, val_loader = get_dataset(opt)

# print(len(train_loader.dataset))
# print(len(val_loader.dataset))

############### Overall Test ###############
# WORLD_SIZE = dist.get_world_size()
# RANK = dist.get_rank()
# torch.cuda.set_device(RANK)
# opt['WORLD_SIZE'] = WORLD_SIZE
# opt = EasyDict(opt)

# model = get_model(opt)
# train_loader, val_loader = get_dataset(opt)

# data = next(iter(train_loader))
# if isinstance(data['image'], list):
#     image = [img.to(RANK) for img in data['image']]
#     label = [lab.to(RANK) for lab in data['label']]
# else:
#     image, label = data['image'].to(RANK), data['label'].to(RANK)

# output = model.forward(image)
# print(len(output))
# for o in output:
#     print(o.shape)
