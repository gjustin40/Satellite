from torch import nn

from torch.nn import functional as F

import torch
import math

class MetricTracker():
    """
    원하는 Metric을 전부 계산하고 Dict형태로 출력하는 클래스
    - Train시 Score는 첫 번쨰 Interval부터 축척된 평균값임
    - Resume시 checkpoint까지 저장된 평균값에서부터 다시 시작
    """
    def __init__(self, opt):
        self.world_size = opt.WORLD_SIZE
        self.metrics = opt.CHECKPOINT.METRICS
        self.result = {m:0 for m in opt.CHECKPOINT.METRICS}

        # self.avg = {m:0 for m in opt.CHECKPOINT.METRICS}
        self.avg = None
        self.avg_que = [] # [t-1, t] 평균
        
    def get_metric(self, input, target, rank):
        """
        1 interval 당 Metric값 계산 (Batch 및 Multi-GPU 평균)

        input: [B, H, W] or [B, 1, H, W]
        target: [B, H, W] or [B, 1, H, W]

        return: {
            'metric1': score1,
            'metric2': score2
        }
        """
        for metric in self.metrics:
            score = eval(metric)(input, target).to(rank)
            dist.all_reduce(score, op=dist.ReduceOp.SUM)

            self.result[metric] = (score / opt.WORLD_SIZE).item()


    def _get_avg(self, result):
        if self.avg is None:
            self.avg = result
        else:
            for m, score in result.items():
                self.avg[m] = (self.avg[m] + score) / 2


# # https://github.com/pytorch/pytorch/issues/1249
def Dice(input, target):
    '''
    Binary만 계산
    input: [B, C, H, W]
    target: [B, H, W] or [B, 1, H, W]
    '''
    num_in_target = input.shape[0]
    
    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    score = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return score.mean()

def IoU(input, target):
    """IoU calculation """
    num_in_target = input.size(0)

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    # intersection = (pred*truth).long().sum(1).data.cpu()[0]
    intersection = (pred * truth).sum(1)
    
    # union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection
    union = pred.sum(1) + truth.sum(1) - intersection

    score = (intersection + 1e-15) / (union + 1e-15)

    return score.mean()


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    with open('../configs/config.yaml', "r") as f:
        opt = yaml.safe_load(f)
    opt = EasyDict(opt)

    train_metric = MetricTracker(opt)
    print(train_metric.results)
    print(eval('Dice'))
# class MetricTracker(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count



# if __name__ == '__main__':
#     a = torch.ones((5,1,256,256))
#     b = torch.ones((5,256,256))

#     print(dice(a,b))
