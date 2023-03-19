# from easydict import EasyDict
# from models import create_model

# opt = {
# 'MODEL_NAME': 'resnet18',
# 'NUM_CLASSES': 10,
# 'PRETRAINED': False,
# 'RESUME': False,
# 'LR': 0.01,
# 'EPOCH': 4,
# 'VAL_EPOCH': 2,
# 'WORLD_SIZE': 1
# }
# opt = EasyDict(opt)


# model = create_model(opt, 0)
# model

from models.networks import build_network
import torch
net = build_network('resnet18', num_classes=10)
inp = torch.randn(1,3,32,32)
out = net(inp)
print(out)