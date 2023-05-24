import torch
from torch import nn
from torch.nn import functional as F

class BCELoss(nn.Module):
    def __init__(self, weight=None, aux_weights=[1, 0.4, 0.4]):
        super().__init__()
        self.weight = weight
        self.aux_wegiths = aux_weights
        self.criterion = nn.BCEWithLogitsLoss()

    def _forward(self, preds, labels):
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        # (C == 1) in binary segmentation

        if (preds.dim() == 4) and (labels.dim() == 3):
            preds = preds.squeeze(1)
        elif (preds.dim() == 3) and (labels.dim() == 4):
            labels = labels.squeeze(1)

        return self.criterion(preds, labels)

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class MSELoss(nn.Module):
    def __init__(self, weight=None, aux_weights=[1, 0.4, 0.4]):
        super().__init__()
        self.weight = weight
        self.aux_wegiths = aux_weights
        self.criterion = nn.MSELoss()

    def _forward(self, preds, labels):
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        # (C == 1) in binary segmentation

        if (preds.dim() == 4) and (labels.dim() == 3):
            preds = preds.squeeze(1)
        elif (preds.dim() == 3) and (labels.dim() == 4):
            labels = labels.squeeze(1)

        return self.criterion(preds, labels)

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)
    

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, aux_weights=[0.4, 1]):
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def _forward(self, preds, labels):
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]

        if (preds.dim() == 4) and (labels.dim() == 4):
            labels = labels.squeeze(1)
        assert preds.dim() == 4, f"The dim of model's output should be 4, current {preds.dim()}"
        assert labels.dim() == 3, f"The dim of label should be 3, current {labels.dim()}"

        return self.criterion(preds, labels.long())

    def forward(self, preds, labels):
        if isinstance(preds, list):
            return sum([w * self._forward(pred, labels.long()) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class DiceLoss2(nn.Module):
    def __init__(self, smooth=1.0, aux_weights=[1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.smooth = smooth
        self.aux_weights = aux_weights

    def _forward(self, preds, labels):
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_batch = preds.size(0)

        preds_ = F.sigmoid(preds)

        preds_ = preds.view(num_batch, -1)
        labels_ = labels.view(num_batch, -1)

        intersection = (preds_ * labels_).sum(1)

        dice_score = (2. * intersection + self.smooth) /(preds_.sum(1) + labels_.sum(1) + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss.mean()

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)

class DiceLoss(nn.Module):
    def __init__(self, smooth=0.0, aux_weights=[1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.smooth = smooth
        self.aux_weights = aux_weights

    def _forward(self, preds, labels):
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_batch = preds.size(0)

        preds_ = torch.sigmoid(preds) > 0.5

        preds_ = preds.view(-1)
        labels_ = labels.view(-1)

        intersection = (preds_ * labels_).sum()

        dice_score = (2. * intersection + self.smooth) /(preds_.sum() + labels_.sum() + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)

class CriterionPixelWise(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ignore_index = ignore_index
        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        # if not reduce:
        #     print("disabled the reduce.")

    def forward(self, preds_S, preds_T):
        # preds_T[0] is the seg logit
        preds_T.detach()
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.permute(0,2,3,1).contiguous().view(-1,C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        return loss


class CriterionPairWise(nn.Module):
    def __init__(self, scale=0.5):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWise, self).__init__()
        self.criterion = self.sim_dis_compute
        self.scale = scale

    def L2(self, f_):
        return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

    def similarity(self, feat):
        feat = feat.float()
        tmp = self.L2(feat).detach()
        feat = feat/tmp
        feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
        return torch.einsum('icm,icn->imn', [feat, feat])

    def sim_dis_compute(self, f_S, f_T):
        sim_err = ((self.similarity(f_T) - self.similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
        sim_dis = sim_err.sum()
        return sim_dis

    def forward(self, preds_S, preds_T):
        assert preds_S.shape == preds_T.shape, 'Studnet and Teacher output shape should be same'
        
        preds_T.detach()

        total_w, total_h = preds_T.shape[2], preds_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(preds_S), maxpool(preds_T))
        return loss


if __name__ == '__main__':
    fn = DiceLoss()
    inp = torch.randn(1,1,512,512)
    out = torch.randn(1,1,512,512)

    print(fn(inp, out))