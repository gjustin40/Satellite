
from .base import BaseModel
from .networks import *
from utils import *
import segmentation_models_pytorch as smp
import torch.nn as nn

""" Description About UperNetKD4 Model
    - EXP-4
    - Normal Method of KD to BEiT-Adapter
"""
class UperNetKD4Model(BaseModel):
    def __init__(self, opt):
        super(UperNetKD4Model, self).__init__(opt=opt)
        self.teacher_net, self.net = self._get_network()
        
        # Loss Functions
        self.loss_CE = CrossEntropyLoss()
        # self.loss_mse = MSELoss()
        self.loss_pairwise = CriterionPairWise() # 그냥 MSE인데 MaxPooling 적용한거
        self.loss_pixelwise = CriterionPixelWise()


    def forward(self, image):
        # image = [aug_optical['image'].float(), aug_sar['image'].float()]
        # optical_image, sar_image
        with torch.no_grad():
            self.teacher_net.eval()
            teacher_outputs = self.teacher_net(image[0])


        # output of teacher net
        # out_list = features + decode + [out]
        # [4 feature maps] + [1 output] = 5 maps
        # torch.Size([1, 1024, 128, 128]) # 1/4
        # torch.Size([1, 1024, 64, 64]) # 1/8
        # torch.Size([1, 1024, 32, 32]) # 1/16 feature 3
        # torch.Size([1, 1024, 16, 16]) # 1/32 feature 4
        # torch.Size([1, 1024, 128, 128]) # decode
        # torch.Size([1, 2, 512, 512]) # 1/1 (B, C, 512, 512) # out

        # output of net
        # out_list = features + decode + [out]
        # [4 feature maps] + [1 output] = 5 maps
        # torch.Size([1, 1024, 128, 128]) # 1/4
        # torch.Size([1, 1024, 64, 64]) # 1/8
        # torch.Size([1, 1024, 32, 32]) # 1/16 feature 3
        # torch.Size([1, 1024, 16, 16]) # 1/32 feature 4
        # torch.Size([1, 1024, 128, 128]) # decode
        # torch.Size([1, 2, 512, 512]) # 1/1 (B, C, 512, 512) # out


        return  [teacher_outputs] + self.net(image[1])

    def _get_network(self):
        teacher_net = get_network(
            network_name=self.opt.MODEL.TEACHER_NETWORK_NAME,
            in_chans=self.opt.MODEL.IN_CHANNELS,
            num_classes=self.opt.MODEL.NUM_CLASSES)
        
        teacher_checkpoint = torch.load(self.opt.MODEL.TEACHER_NETWORK_lOAD_PATH, map_location='cpu')
        teacher_net.load_state_dict(teacher_checkpoint['state_dict'])
        if self.rank == 0:
            print('Loading Teacher Model checkpoint.....')

        net = get_network(
            network_name=self.opt.MODEL.NETWORK_NAME,
            in_chans=self.opt.MODEL.IN_CHANNELS,
            num_classes=self.opt.MODEL.NUM_CLASSES)
        net.apply(init_weights)
        
        if self.opt.MODEL.PRETRAINED_PATH:
            checkpoint = torch.load(self.opt.MODEL.PRETRAINED_PATH, map_location='cpu')
            net.load_state_dict(checkpoint['state_dict'])
            if self.rank == 0:
                print('Loading pretrained Model checkpoint.....')
        return [self._wrap_ddp(teacher_net), self._wrap_ddp(net)]


    def get_loss(self, output, label):

        ########## label
        # torch.Size([1, 1, 512, 512])

        # output of teacher net
        # out_list = features + decode + [out]
        # [4 feature maps] + [1decode ] + [1 output] = 6 maps
        # torch.Size([1, 1024, 128, 128]) # 1/4
        # torch.Size([1, 1024, 64, 64]) # 1/8
        # torch.Size([1, 1024, 32, 32]) # 1/16 feature 3
        # torch.Size([1, 1024, 16, 16]) # 1/32 feature 4
        # torch.Size([1, 1024, 128, 128]) # decode
        # torch.Size([1, 2, 512, 512]) # 1/1 (B, C, 512, 512) # out

        # output of net
        # out_list = features + decode + [out]
        # [4 feature maps] + [1 output] = 5 maps
        # torch.Size([1, 1024, 128, 128]) # 1/4
        # torch.Size([1, 1024, 64, 64]) # 1/8
        # torch.Size([1, 1024, 32, 32]) # 1/16 feature 3
        # torch.Size([1, 1024, 16, 16]) # 1/32 feature 4
        # torch.Size([1, 1024, 128, 128]) # decode
        # torch.Size([1, 2, 512, 512]) # 1/1 (B, C, 512, 512) # out

        # 1. feature map loss
        # PairWise
        # 2. teacher-student output loss
        # Pixelwise
        # 3. student GT loss
        # CELoss
        # [[6], 6]
        T_features, S_features = output[0][:4], output[1:5]
        T_output, S_output = output[0][-1], output[-1]
        
        # pair_loss = sum([self.loss_pairwise(s, t)  for s, t in zip(S_features, T_features)])
        # pixel_loss = self.loss_pixelwise(S_output, T_output)
        gt_loss = self.loss_CE(S_output, label)

        # loss = pair_loss + pixel_loss + gt_loss
        # loss = pair_loss + gt_loss
        # loss = pixel_loss + gt_loss
        loss = gt_loss

        # if self.rank == 0:
        #     print('------------------Start---------------------')
        #     print('self.pixel_loss')
        #     print('S_output', S_output.shape)
        #     print('T_output', T_output.shape)
        #     print('-------------')

        #     print('self.loss_CE')
        #     print('S_output', S_output.shape)
        #     print('label', label.shape)
        #     print('--------------')

        #     print('self.loss_pairwise')
        #     for a,b in zip(S_features, T_features):
        #         print('S_features / T_features', a.shape, b.shape)

        return loss

    # def get_params(self):
    #     return layer_decay_optimizer_constructor(self.opt, self.net)

    # def predict(self, output, label)