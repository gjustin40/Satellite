# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from .beit import BEiT
from .modules.adapter_modules import SpatialPriorModule, deform_inputs
from .modules.adapter_modules import InteractionBlockWithCls as InteractionBlock
_logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings(action='ignore')

class BEiTAdapter(BEiT):
    def __init__(self, pretrain_size=512, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., cffn_ratio=0.25, deform_ratio=1.0, with_cffn=True,
                 interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23], [12, 17], [18, 23]], 
                 add_vit_feature=True, with_cp=True, *args, **kwargs):

        super().__init__(init_values=init_values, with_cp=with_cp, *args, **kwargs)

        # self.num_classes = 80
        # self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.flags = [i for i in range(-1, self.num_block, self.num_block // 4)][1:]
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim
        in_chans = self.in_chans
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(in_chans=in_chans, inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=True if i == len(interaction_indexes) - 1 else False,
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])

        self.up_opt = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.up_sar = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1_sar = nn.BatchNorm2d(embed_dim)
        self.norm2_sar = nn.BatchNorm2d(embed_dim)
        self.norm3_sar = nn.BatchNorm2d(embed_dim)
        self.norm4_sar = nn.BatchNorm2d(embed_dim)

        self.norm3_opt = nn.BatchNorm2d(embed_dim)
        self.norm4_opt = nn.BatchNorm2d(embed_dim)

        self.up_opt.apply(self._init_weights)
        self.up_sar.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        # self.frozen_stages = frozen_stages
        # stage0 self.patch_embed + self.cls_token + self.pos_drop -> dropout
        # stage1 self.spm + self._add_level_embed
        # stage2 self.interactions
        # stage3 self.up + self.norm1 ~ 4
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4


    def forward2(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        cls = self.cls_token.expand(bs, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.pos_embed is not None:
            pos_embed = self._get_pos_embed(self.pos_embed, H, W)
            x = x + pos_embed
        x = self.pos_drop(x)

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.blocks[indexes[0]:indexes[-1] + 1],
                              deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())



    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        cls = self.cls_token.expand(bs, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        if self.pos_embed is not None:
            pos_embed = self._get_pos_embed(self.pos_embed, H, W)
            x = x + pos_embed
        x = self.pos_drop(x)

        # Interaction
        outs_opt = list()
        outs_sar = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            if i < 2:
                x, c, cls = layer(x, c, cls, self.blocks[indexes[0]:indexes[-1] + 1],
                                deform_inputs1, deform_inputs2, H, W)
                outs_opt.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())
                outs_sar.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

            elif i == 2:
                x_opt, c_opt, cls_opt = x.clone(), c.clone(), cls.clone()
                x_sar, c_sar, cls_sar = x.clone(), c.clone(), cls.clone()

                x_opt, c_opt, cls_opt = layer(x_opt, c_opt, cls_opt, self.blocks[indexes[0]:indexes[-1] + 1],
                                deform_inputs1, deform_inputs2, H, W)
                outs_opt.append(x_opt.transpose(1, 2).view(bs, dim, H, W).contiguous())

            elif i == 3:
                x_opt, c_opt, cls_opt = layer(x_opt, c_opt, cls_opt, self.blocks[indexes[0]:indexes[-1] + 1],
                                deform_inputs1, deform_inputs2, H, W)
                outs_opt.append(x_opt.transpose(1, 2).view(bs, dim, H, W).contiguous())
                
            elif i == 4:
                x_sar, c_sar, cls_sar = layer(x_sar, c_sar, cls_sar, self.blocks[indexes[0]:indexes[-1] + 1],
                                deform_inputs1, deform_inputs2, H, W)
                outs_sar.append(x_sar.transpose(1, 2).view(bs, dim, H, W).contiguous())

            elif i == 5:
                x_sar, c_sar, cls_sar = layer(x_sar, c_sar, cls_sar, self.blocks[indexes[0]:indexes[-1] + 1],
                                deform_inputs1, deform_inputs2, H, W)
                outs_sar.append(x_sar.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        # sar
        c_sar2 = c_sar[:, 0:c2.size(1), :]
        c_sar3 = c_sar[:, c_sar2.size(1):c_sar2.size(1) + c3.size(1), :]
        c_sar4 = c_sar[:, c_sar2.size(1) + c_sar3.size(1):, :]

        c_sar2 = c_sar2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c_sar3 = c_sar3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c_sar4 = c_sar4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c_sar1 = self.up_sar(c_sar2) + c1

        # optical
        c_opt2 = c_opt[:, 0:c2.size(1), :]
        c_opt3 = c_opt[:, c_opt2.size(1):c_opt2.size(1) + c3.size(1), :]
        c_opt4 = c_opt[:, c_opt2.size(1) + c_opt3.size(1):, :]

        c_opt2 = c_opt2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c_opt3 = c_opt3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c_opt4 = c_opt4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c_opt1 = self.up_opt(c_opt2) + c1


        # Split & Reshape
        # c2 = c[:, 0:c2.size(1), :]
        # c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        # c4 = c[:, c2.size(1) + c3.size(1):, :]

        # c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        # c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        # c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        # c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x_sar1, x_sar2, x_sar3, x_sar4 = outs_sar
            x_sar1 = F.interpolate(x_sar1, scale_factor=4, mode='bilinear', align_corners=False)
            x_sar2 = F.interpolate(x_sar2, scale_factor=2, mode='bilinear', align_corners=False)
            x_sar4 = F.interpolate(x_sar4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c_sar1, c_sar2, c_sar3, c_sar4 = c_sar1 + x_sar1, c_sar2 + x_sar2, c_sar3 + x_sar3, c_sar4 + x_sar4

            _, _, x_opt3, x_opt4 = outs_opt
            x_opt4 = F.interpolate(x_opt4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c_opt3, c_op4 = c_opt3 + x_opt3, c_opt4 + x_opt4

        # if self.add_vit_feature:
        #     x1, x2, x3, x4 = outs
        #     x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        #     x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        #     x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
        #     c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1_sar = self.norm1_sar(c_sar1)
        f2_sar = self.norm2_sar(c_sar2)
        f3_sar = self.norm3_sar(c_sar3)
        f4_sar = self.norm4_sar(c_sar4)

        f3_opt = self.norm3_opt(c_opt3)
        f4_opt = self.norm4_opt(c_opt4)

        return [[f1_sar, f2_sar], [f3_opt, f4_opt], [f3_sar, f4_sar]]
        # f1 = self.norm1(c1)
        # f2 = self.norm2(c2)
        # f3 = self.norm3(c3)
        # f4 = self.norm4(c4)

        # return [f1, f2, f3, f4]


if __name__ == '__main__':
    model = BEiTAdapter().to('cuda:4')
    inp = torch.Tensor(1,3,512,512).to('cuda:4')
    output = model(inp)
    print(len(output))