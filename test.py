import yaml
import argparse
from easydict import EasyDict
import pickle

import torch
from tqdm import tqdm
import numpy as np

from models import get_network
from datasets import get_dataset
from utils.metrics import mDice

# parser = argparse.ArgumentParser(description='Train a segmentor')
# parser.add_argument('config', help='train config file path')
# parser.add_argument('--gpu_id', default=0, help='ID of GPU for test')
# args = parser.parse_args()



# with open('args.config', "r") as f:
#     opt = yaml.safe_load(f)
#     opt = EasyDict(opt)

with open('/home/yh.sakong/github/Satellite/configs/config_beit_adapter_upernet_kd_test.yaml', "r") as f:
    opt = yaml.safe_load(f)
    opt = EasyDict(opt)

network_name = opt.MODEL.NETWORK_NAME
in_channels = opt.MODEL.IN_CHANNELS
num_classes = opt.MODEL.NUM_CLASSES
checkpoint_path = opt.TEST.TEST_PATH
# device = f'cuda:{args.gpu_id}'
device = f'cuda:0'

net = get_network(network_name, in_chans=in_channels, num_classes=num_classes)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['state_dict']
net.load_state_dict(state_dict)
net.to(device)

test_loader = get_dataset(opt)
tbar = tqdm(test_loader, dynamic_ncols=True, desc="Validation")

result = {str(k):0 for k in np.arange(0, 1, 0.1)}
results = []
results_lab = []
with torch.no_grad():
    net.eval()

    for data in tbar:
        image, label = data['image'].to(device), data['label']
        output = net(image)
        output = torch.sigmoid(output[-1].cpu())
        pred = (output > 0.3).float()
        results.append(np.array(pred[0][0]))
        results_lab.append(np.array(label[0]))
        # break
        # for thresh in np.arange(0, 1, 0.1):
        #     dice = Dice(pred, label)
        #     result[str(thresh)] += dice
with open('aa.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('bb.pkl', 'wb') as f:
    pickle.dump(results_lab, f)

for k,v in result.items():
    print(f'threshold: {k} score: {v/len(tbar)}')
