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
            preds_ = preds.squeeze(1)
        elif (preds.dim() == 3) and (labels.dim() == 4):
            labels = labels.squeeze(1)

        return self.criterion(preds_, labels)

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
            preds_ = preds.squeeze(1)
        elif (preds.dim() == 3) and (lables.dim() == 4):
            preds_ = label.squeeze(1)

        return self.criterion(preds_, labels)

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
        return self.criterion(preds, labels)

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


if __name__ == '__main__':
    fn = DiceLoss()
    inp = torch.randn(1,1,512,512)
    out = torch.randn(1,1,512,512)

    print(fn(inp, out))