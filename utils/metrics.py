from torch import nn

from torch.nn import functional as F

import torch
import math

class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# https://github.com/pytorch/pytorch/issues/1249
def dice(input, target):
    num_in_target = input.shape[0]
    
    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    score = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return score.mean().item()

def iou(input, target):
    """IoU calculation """
    num_in_target = input.size(0)

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    # intersection = (pred*truth).long().sum(1).data.cpu()[0]
    intersection = (pred * truth).sum(1)
    
    # union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection
    union = pred.sum(1) + truth.sum(1) - intersection

    score = (intersection + 1e-15) / (union + 1e-15)

    return score.mean().item()

if __name__ == '__main__':
    a = torch.ones((5,1,256,256))
    b = torch.ones((5,256,256))

    print(dice(a,b))
