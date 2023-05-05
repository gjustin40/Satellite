import torch
import torch.nn as nn
import torch.nn.functional as F

from .beit_adapter import BEiTAdapter
from .uper_head import UPerHead

class BEiTAdapterUperNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1):
        super(BEiTAdapterUperNet, self).__init__()
        self.backbone = BEiTAdapter(
            pretrain_size=512, in_chans=in_chans, conv_inplane=64, n_points=4, deform_num_heads=16,
            init_values=1e-06, cffn_ratio=0.25, deform_ratio=0.5, with_cffn=True,
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]], 
            add_vit_feature=True, with_cp=True, 
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            use_abs_pos_emb=False, use_rel_pos_bias=True, drop_path_rate=0.3)

        self.decode_head = UPerHead(
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

    def forward(self, x):
        features = self.backbone(x)
        out = self.decode_head(features)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        out_list = features + [out]
        
        return out_list