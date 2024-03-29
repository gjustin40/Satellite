import torch
import torch.distributed as dist

import numpy as np
import torch

class MetricTracker():
    """
    원하는 Metric을 전부 계산하고 Dict형태로 출력하는 클래스
    - Train시 Score는 첫 번쨰 Interval부터 축척된 평균값임
    - Resume시 checkpoint까지 저장된 평균값에서부터 다시 시작
    - Resume시 self.sum = checkpoint['avg']
    """
    def __init__(self, opt):
        self.opt = opt
        self.world_size = opt.WORLD_SIZE
        self.metrics = opt.CHECKPOINT.METRICS
        self.num_classes = opt.MODEL.NUM_CLASSES

        # Because of Binary Segmentation (1 == 2)
        # self.num_classes = 2 if opt.MODEL.NUM_CLASSES == 1 else opt.MODEL.NUM_CLASSES
        
        # Resume모드일 땐 기존 avg를 이어서 사용하기 때문에 interval=1 로 시작
        # self.interval = 0 if opt.MODEL.RESUME_PATH is None else 1
        self.sum      = {m:0 for m in opt.CHECKPOINT.METRICS}
        self.avg      = {m:0 for m in opt.CHECKPOINT.METRICS}

        # metric_classes = 2 if opt.MODEL.NUM_CLASSES == 1 else opt.MODEL.NUM_CLASSES
        n_classes = 2 if self.num_classes == 1 else self.num_classes
        self.total_area_intersect = torch.zeros((n_classes, ), dtype=torch.float64)
        self.total_area_pred      = torch.zeros((n_classes, ), dtype=torch.float64)
        self.total_area_label     = torch.zeros((n_classes, ), dtype=torch.float64)
        self.total_area_union     = torch.zeros((n_classes, ), dtype=torch.float64)
        
        self.result = {m:0 for m in opt.CHECKPOINT.METRICS}

    def get(self, output, label, rank):
        output, label = self._preprocessing_inputs(output, label)
        
        if self.num_classes == 1: # Sigmoid
            pred = (torch.sigmoid(output) > self.opt.CHECKPOINT.THRESHOLD).float()
            area_intersect, area_pred, area_label, area_union \
            = intersect_and_union(pred, label, self.num_classes+1) # [background, foreground]
        else: 
            pred = torch.argmax(output, dim=1)
            area_intersect, area_pred, area_label, area_union \
            = intersect_and_union(pred, label, self.num_classes)
        
        dist.all_reduce(area_intersect.to(rank), op=dist.ReduceOp.SUM)
        dist.all_reduce(area_pred.to(rank), op=dist.ReduceOp.SUM)
        dist.all_reduce(area_label.to(rank), op=dist.ReduceOp.SUM)
        dist.all_reduce(area_union.to(rank), op=dist.ReduceOp.SUM)

        self.total_area_intersect += area_intersect.cpu()
        self.total_area_pred      += area_pred.cpu()
        self.total_area_label     += area_label.cpu()
        self.total_area_union     += area_union.cpu()

        for metric in self.metrics:
            if metric == 'mIoU':
                 # tensor([score1, score2, ...])
                score = self.total_area_intersect / self.total_area_union
                self.result[metric] = score[1].item()
                # self.result[metric] = score[1].item() # 수정 필요
            elif metric == 'mDice':
                # tensor([score1, score2, ...])
                score = (2*self.total_area_intersect + 1) / (self.total_area_pred + self.total_area_label + 1)
                self.result[metric] = score[1].item()
                # self.result[metric] = score # 수정 필요

        # 수정 필요
        return self.result


    def _preprocessing_inputs(self, output, label):
        """To make same attribute about 'output' and 'label' data

        Args:
        output (ndarray | Tensor, [B, C, H, W]): Array of prediction segmentation maps with class index.
        label (ndarray | Tensor): Array of GT segmentation maps with class index.

        Returns:
        output (Tensor, [B, H, W] | [B, C, H, W]): preprocessed output data
        label  (Tensor, [B, H, W]): preprocessed label data
        """
        # To make same attribute about 'output' and 'label' data
        # CPU/GPU, shape, types, etc

        # Usually Output of model should be list (ex. [feature1, feature2, output])
        # So list elements will be last output of model
        if isinstance(output, list):
            output = output[-1]
        if isinstance(label, list):
            label = label[-1]
        
        output = output.detach().cpu()
        label = label.detach().cpu()
        
        if (self.num_classes == 1) and (output.dim() == 4):
            output = output.squeeze(1)

        return output, label

def intersect_and_union(pred, label, num_classes):
    
    """Calculate Intersection and Union.

    Args:
    pred (ndarray | Tensor): Array of prediction segmentation maps with class index.
    label (ndarray | Tensor): Array of GT segmentation maps with class index.
    num_classes (int): Number of categories(class index).

    Shape:
    - pred: :math:`(H, W)`. or :math:`(B, H, W)`
    - label: :math:`(H, W)`. or :math:`(B, H, W)`

    Returns:
    torch.Tensor: The intersection of prediction and ground truth
    histogram on all classes.
    torch.Tensor: The prediction histogram on all classes.
    torch.Tensor: The ground truth histogram on all classes.
    torch.Tensor: The union of prediction and ground truth histogram on
    all classes.
    """
    assert pred.shape == label.shape, f'pred & label shapes should be same. {pred.shape} != {label.shape}'

    pred  = pred.detach().cpu()
    label = label.detach().cpu()
    
    intersect = pred[pred == label]
    
    # torch.histc: 각 class별 histogram 계산(== 개수)
    area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=(num_classes-1))
    area_pred      = torch.histc(pred.float(), bins=(num_classes), min=0, max=(num_classes-1))
    area_label     = torch.histc(label.float(), bins=(num_classes), min=0, max=(num_classes-1))
    area_union     = area_pred + area_label - area_intersect
    
    return area_intersect, area_pred, area_label, area_union


def iou(pred, label, num_classes, name_list=None):
    # calculate single iou
    area_intersect, area_pred, area_label, area_union \
    = intersect_and_union(pred, label, num_classes)
    
    return area_intersect / area_union


def dice(pred, label, num_classes, name_list=None):
    # calculate single dice
    area_intersect, area_pred, area_label, area_union \
    = intersect_and_union(pred, label, num_classes)
    
    return (2*area_intersect +1) / (area_pred + area_label + 1)



def get_metric_from_area(
    area_intersect,
    area_pred,
    area_label,
    area_union,
    metrics=['mIoU'],
    beta=1
    ):
    
    if isinstance(metrics, str):
        metrics = [metrics]
        
    metric_list = ['mIoU', 'mDice', 'mFscore']
    assert set(merics).issubset(metric_list), f'we only support ["mIoU", "mDice", "mFscore"]'
    
    all_acc = area_intersect.sum() / area_label.sum()
    acc     = area_intersect / area_label
    metric_result = OrderedIdct({'aAcc': all_acc, 'Acc': acc})
    if 'mIoU' in metrics:
        iou = iou = area_intersect / area_union
    elif 'mDice' in metrics:
        dice = (2*area_intersect) / (area_pred + area_label)
    elif 'mFscore' in metrics:
        pass




# # https://github.com/pytorch/pytorch/issues/1249
def mDice(input, target):
    num_in_target = input.shape[0]
    
    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    score = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return score.mean()

def Dice2(input, target):
    num_in_target = input.shape[0]
    
    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    pred = torch.where(pred == 1, 1, 0)
    truth = torch.where(truth == 1, 1, 0)

    intersection = (pred * truth).sum(1)

    score = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return score.mean()

def mIoU(input, target):
    num_in_target = input.size(0)

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    # intersection = (pred*truth).long().sum(1).data.cpu()[0]
    intersection = (pred * truth).sum(1)
    
    # union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection
    union = pred.sum(1) + truth.sum(1) - intersection

    score = (intersection + 1e-15) / (union + 1e-15)

    return score.mean()

def IoU2(input, target):
    num_in_target = input.size(0)

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    pred = torch.where(pred == 1, 1, 0)
    truth = torch.where(truth == 1, 1, 0)

    # intersection = (pred*truth).long().sum(1).data.cpu()[0]
    intersection = (pred * truth).sum(1)
    
    # union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection
    union = pred.sum(1) + truth.sum(1) - intersection

    score = (intersection + 1e-15) / (union + 1e-15)

    return score.mean()