import torch
import torch.nn as nn
import torch.nn.functional as F

from .beit_adapter2 import BEiTAdapter
from .uper_head2 import UPerHead2

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class FeatureFusionModuleSCSE_V2(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super().__init__()
        
        self.scse_1 = SCSEModule(in_chan)
        self.scse_2 = SCSEModule(in_chan)
        self.convblk = ConvBNReLU(in_chan*2, out_chan, ks=1, stride=1, padding=0)
        self.scse = SCSEModule(out_chan)
        self.init_weight()

    def forward(self, fsp, fcp):
        fsp = self.scse_1(fsp)
        fcp = self.scse_2(fcp)
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        feat_out = self.scse(feat)
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class BEiTAdapterUperNetKD(nn.Module):
    def __init__(self, in_chans=3, num_classes=1):
        super(BEiTAdapterUperNetKD, self).__init__()
        self.backbone = BEiTAdapter(
            pretrain_size=512, in_chans=in_chans, conv_inplane=64, n_points=4, deform_num_heads=16,
            init_values=1e-06, cffn_ratio=0.25, deform_ratio=0.5, with_cffn=True,
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23], [12, 17], [18, 23]], 
            # interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23], [24, 29], [30, 35]], 
            add_vit_feature=True, with_cp=True, 
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            use_abs_pos_emb=False, use_rel_pos_bias=True, drop_path_rate=0.3)

        self.decode_opt = UPerHead2(
            in_channels=[1024, 1024, 1024, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=1024,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))

        self.decode_sar = UPerHead2(
            in_channels=[1024, 1024, 1024, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=1024,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))

        self.ffm = FeatureFusionModuleSCSE_V2(in_chan=1024, out_chan=1024)
        self.ffm_seg_head = nn.Conv2d(1024, num_classes, kernel_size=1)
        

    def forward(self, x):
        features = self.backbone(x)
        features_opt = features[0] + features[1]
        features_sar = features[0] + features[2]

        decode_opt, out_opt = self.decode_opt(features_opt)
        decode_sar, out_sar = self.decode_sar(features_sar)

        out_opt = F.interpolate(out_opt, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        out_sar = F.interpolate(out_sar, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape

        out_combine = self.ffm_seg_head(self.ffm(decode_sar, decode_opt))
        out_combine = F.interpolate(out_combine, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape

        # features_opt : 2 maps
        # torch.Size([1, 1024, 32, 32]) # 1/16
        # torch.Size([1, 1024, 16, 16]) # 1/32

        # features_sar : 4 maps
        # torch.Size([1, 1024, 128, 128]) # 1/4
        # torch.Size([1, 1024, 64, 64]) # 1/8
        # torch.Size([1, 1024, 32, 32]) # 1/16
        # torch.Size([1, 1024, 16, 16]) # 1/32

        # decode_opt : feature of opt decoder before seg head
        # torch.Size([1, 1024, 128, 128]) # 1/4

        # decode_sar : feature of sar decoder before seg head
        # torch.Size([1, 1024, 128, 128]) # 1/4

        # out_opt : output of optical branch [B, C, H, W] (after decode_opt)
        # torch.Size([1, 1, 512, 512])

        # out_combine : output of network --> goal
        # torch.Size([1, 1, 512, 512])

        return [
            features_opt[-2:], 
            features_sar, 
            decode_opt, 
            out_opt, 
            out_sar,
            out_combine
        ]