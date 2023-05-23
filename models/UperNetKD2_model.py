
from .base import BaseModel
from .networks import *
from utils import *
import segmentation_models_pytorch as smp
import torch.nn as nn

"""Description About UperNetKD2 Model

"""
class UperNetKD2Model(BaseModel):
    def __init__(self, opt):
        super(UperNetKD2Model, self).__init__(opt=opt)
        self.teacher_net, self.net = self._get_network()
        
        # Loss Functions
        self.loss_CE = CrossEntropyLoss()
        self.loss_mse = MSELoss()
        self.pixelwise = CriterionPixelWise()

        # Metric Functions


    def forward(self, image):
        # image = [aug_optical['image'].float(), aug_sar['image'].float()]
        # optical_image, sar_image
        with torch.no_grad():
            self.teacher_net.eval()
            teacher_outputs = self.teacher_net(image[0])


        # output of teacher net
        # out_list = features + [out]
        # [4 feature maps] + [1 output] = 5 maps
        # torch.Size([1, 1024, 128, 128]) # 1/4
        # torch.Size([1, 1024, 64, 64]) # 1/8
        # torch.Size([1, 1024, 32, 32]) # 1/16 feature 3
        # torch.Size([1, 1024, 16, 16]) # 1/32 feature 4
        # torch.Size([1, 1024, 128, 128]) # decode
        # torch.Size([1, 1, 512, 512]) # 1/1 (B, C, 512, 512) # out


        # output of net
        # features_opt : 4 maps
        # torch.Size([1, 1024, 32, 32]) # 1/16 -- use
        # torch.Size([1, 1024, 16, 16]) # 1/32 -- use

        # features_sar : 4 maps
        # torch.Size([1, 1024, 128, 128]) # 1/4
        # torch.Size([1, 1024, 64, 64]) # 1/8
        # torch.Size([1, 1024, 32, 32]) # 1/16
        # torch.Size([1, 1024, 16, 16]) # 1/32

        # decode_opt : 1 map
        # torch.Size([1, 1024, 128, 128])

        # out_opt : output of optical branch [B, C, H, W]
        # torch.Size([1, 1, 512, 512])

        # out_sar : output of sar branch [B, C, H, W]
        # torch.Size([1, 1, 512, 512])

        # out_combine : output of net which is goal
        # torch.Size([1, 1, 512, 512])

        # [[4], [4], [4], 1, 1, 1, 1]

        return  [teacher_outputs[-4:]] + self.net(image[1])

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
        # [[4], [4], [4], 1, 1, 1, 1, 1]

        ########## label
        # torch.Size([1, 1, 512, 512])

        ########## output list
        # [0] GT of teacher net, list
        # torch.Size([1, 1024, 32, 32]) # 1/16
        # torch.Size([1, 1024, 16, 16]) # 1/32
        # torch.Size([1, 1024, 128, 128]) # 1/32
        # torch.Size([1, 2, 512, 512]) # 1/1

        # [1] opt branch features, list
        # torch.Size([1, 1024, 32, 32]) # 1/16 -- use
        # torch.Size([1, 1024, 16, 16]) # 1/32 -- use

        # [2] sar branch features, list
        # torch.Size([1, 1024, 128, 128]) # 1/4
        # torch.Size([1, 1024, 64, 64]) # 1/8
        # torch.Size([1, 1024, 32, 32]) # 1/16
        # torch.Size([1, 1024, 16, 16]) # 1/32

        # [3] opt branch decode, tensor
        # torch.Size([1, 1024, 128, 128])

        # [4] opt branch output, tensor
        # torch.Size([1, 2, 512, 512])

        # [5] sar branch output, tensor
        # torch.Size([1, 2, 512, 512])

        # [6] combine output, tensor
        # torch.Size([1, 2, 512, 512])



        ####### Loss process
        # 1. sar branch output loss (original loss)
        # (sar branch output) - (GT of label) --> dice+cross (==BCE)
        
        # 2. opt branch output loss (distillation)
        # (opt branch output) - (GT of teahcer net[-1] with logit) --> dice+cross (==BCE)
        # (opt branch output) - (GT of teacher net[-1] without logit) --> CriterionPixelWise 

        # 3. fused Module loss
        # (combine output) - (GT of label)

        # 4. opt branch features loss (distillation) [features + decode]
        # (opt branch features + decode) - (featurs of GT of teacher net[:2]) --> MSELoss


        # 1. sar branch output loss
        sar_output_loss = self.loss_CE(output[5], label)

        # 2. opt branch output loss (distillation)
        teacher_pred = torch.argmax(output[0][-1], dim=1)
        opt_output_loss_logit = self.loss_CE(output[4], teacher_pred.detach())
        opt_output_loss       = self.pixelwise(output[4], output[0][-1])
        
        # 3. fused loss
        fused_loss = self.loss_CE(output[6], label)

        # 4. opt branch features loss (3 maps = features 2 + decode 1)
        dist_list = output[1] + [output[3]]
        opt_dis_loss = sum([self.loss_mse(opt_f, teacher_f) for opt_f, teacher_f in zip(dist_list, output[0][:3])])

        loss = sar_output_loss + opt_output_loss_logit + opt_output_loss + fused_loss + opt_dis_loss / len(dist_list)

        return loss

    # def get_params(self):
    #     return layer_decay_optimizer_constructor(self.opt, self.net)

    # def predict(self, output, label)