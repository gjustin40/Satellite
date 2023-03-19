from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#########################################################################################################
# RoadNet Loss
def roadnet_loss(surfaces, centers, edges, labels):
    
    def _get_balanced_sigmoid_cross_entropy(gt):
        count_neg = torch.sum(1. - gt)
        count_pos = torch.sum(gt)
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        cost = nn.BCEWithLogitsLoss(size_average=True, reduce=True, pos_weight=pos_weight)
        return cost, 1-beta
    
    weight_surface_side = [0.5, 0.75, 1.0, 0.75, 0.5, 1.0]
    weight_others_side = [0.5, 0.75, 1.0, 0.75, 1.0]
    
    loss_fn = nn.BCEWithLogitsLoss()
    gt_surface = labels[0].unsqueeze(1)
    gt_center = labels[1].unsqueeze(1)
    gt_edge = labels[2].unsqueeze(1)
    surface_fn, surface_beta = _get_balanced_sigmoid_cross_entropy(gt_surface)
    center_fn, center_beta = _get_balanced_sigmoid_cross_entropy(gt_center)
    edge_fn, edge_beta = _get_balanced_sigmoid_cross_entropy(gt_edge)
    
    loss_surface = loss_fn(surfaces[-1], gt_surface) # output, label
    loss_center = loss_fn(centers[-1], gt_center) # output, label
    loss_edge = loss_fn(edges[-1], gt_edge) # output, label
    
    if gt_surface.sum() > 0.0:
        for surface, w in zip(surfaces, weight_surface_side):
            loss_surface += surface_fn(surface, gt_surface) * surface_beta * w
            
    if gt_center.sum() > 0.0:
        for center, w in zip(centers, weight_others_side):
            loss_center += center_fn(center, gt_center) * center_beta * w
            
    if gt_edge.sum() > 0.0:
        for edge, w in zip(edges, weight_others_side):
            loss_edge += edge_fn(edge, gt_edge) * edge_beta * w
            
    total_loss = loss_surface + loss_center + loss_edge
    return total_loss, [loss_surface, loss_center, loss_edge]

#########################################################################################################
# RoadNet Multiclass Loss
def roadnet_multiclass_loss(surfaces, centers, landcovers, labels):
    
    def _get_balanced_sigmoid_cross_entropy(gt):
        count_neg = torch.sum(1. - gt)
        count_pos = torch.sum(gt)
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        cost = nn.BCEWithLogitsLoss(size_average=True, reduce=True, pos_weight=pos_weight)
        return cost, 1-beta
    
    
    weight_surface_side = [0.5, 0.75, 1.0, 0.75, 0.5, 1.0]
    weight_others_side = [0.5, 0.75, 1.0, 0.75, 1.0]
    
    loss_fn = nn.BCEWithLogitsLoss()
    loss_multiclass_fn = nn.CrossEntropyLoss() # Multi-class loss
    gt_surface = labels[0].unsqueeze(1)
    gt_center = labels[1].unsqueeze(1)
    gt_landcover = labels[2] # Multi-class loss [B,H,W]
    
    surface_fn, surface_beta = _get_balanced_sigmoid_cross_entropy(gt_surface)
    center_fn, center_beta = _get_balanced_sigmoid_cross_entropy(gt_center)
    landcover_fn = nn.CrossEntropyLoss()
    
    loss_surface = loss_fn(surfaces[-1], gt_surface) # output, label
    loss_center = loss_fn(centers[-1], gt_center) # output, label
    loss_landcover = loss_multiclass_fn(landcovers[-1], gt_landcover) # output, label / Multi-class loss
    
    if gt_surface.sum() > 0.0:
        for surface, w in zip(surfaces, weight_surface_side):
            loss_surface += surface_fn(surface, gt_surface) * surface_beta * w
            
    if gt_center.sum() > 0.0:
        for center, w in zip(centers, weight_others_side):
            loss_center += center_fn(center, gt_center) * center_beta * w
    
    # Multi-class loss
    if gt_landcover.sum() > 0.0:
        for landcover, w in zip(landcovers, weight_others_side):
            loss_landcover += landcover_fn(landcover, gt_landcover) * w
            
    total_loss = loss_surface + loss_center + loss_landcover
    
    return total_loss, [loss_surface, loss_center, loss_landcover]     
#########################################################################################################
#IoU Loss   
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
#########################################################################################################
# BCE Focal Loss
class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=None, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        loss = - (1 - pt) ** self.gamma * target * torch.log(pt) - \
            pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.alpha:
            loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
            
        return loss
#########################################################################################################
# SSIM Loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
#########################################################################################################   


if __name__ == '__main__':
    from dataset import DeepglobeDatasetMulticlass, DeepglobeDatasetDouble
    import albumentations as A
    from torch.utils.data import DataLoader
    from albumentations.pytorch import ToTensorV2
    import numpy as np
    from networks import create_model
    
    train_img_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_image'
    train_mask_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_segment'
    train_center_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_centerline'
    train_edge_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_edge'
    train_landcover_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_landcover'
    test_img_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/test_image'
    test_mask_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/test_segment'
    test_center_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/test_centerline'
    test_edge_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/test_edge'
    test_landcover_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/test_landcover'
    
    transform_train = A.Compose([
        A.Rotate(limit=40,p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()])
    
    trainset = DeepglobeDatasetDouble(train_img_dir, train_mask_dir, train_edge_dir, transform_train)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=False, num_workers=1)
    
    inputs, labels = iter(trainloader).next()
    print(inputs.shape)
    print(labels[0].shape)
    print(labels[1].shape)
    
    model = create_model(model_name='dtnet', num_classes=1, pre_trained=None)
    with torch.no_grad():
        segments, edges = model(inputs) # surfaces, centers, landcovers
        print(segments.shape)
        print(edges.shape)
        
        loss_fn1 = nn.BCEWithLogitsLoss()
        loss_fn2 = IoULoss()
        loss_fn3 = SSIM(window_size=11, size_average=True)
        loss_fn4 = BCEFocalLoss()
        loss1 = loss_fn1(segments, labels[0].unsqueeze(1))
        loss2 = loss_fn2(segments, labels[0].unsqueeze(1))
        loss3 = loss_fn3(segments, labels[0].unsqueeze(1))
        loss4 = loss_fn4(edges, labels[1].unsqueeze(1))
        loss5 = loss_fn1(edges, labels[1].unsqueeze(1))
        print(loss1, loss2, loss3, loss4, loss5)
        
        total_loss = loss1 + loss2 + (1-loss3) + (2*loss4)
        print(total_loss)
        
        