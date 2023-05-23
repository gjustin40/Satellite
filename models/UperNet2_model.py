
from .base import BaseModel
from .networks import *
from utils import *

class UperNet2Model(BaseModel):
    def __init__(self, opt):
        super(UperNet2Model, self).__init__(opt=opt)
        self.net = self._get_network()

        # Loss Functions
        # self.loss_fn = BCELoss()
        if opt.MODEL.NUM_CLASSES == 1:
            self.loss_fn = BCELoss()
        else:
            self.loss_fn = CrossEntropyLoss()
        # Metric Functions


    def forward(self, image):
        # out_list = features + [out]
        # [4 feature maps] + [1 output] = 5 maps
        # torch.Size([1, 1024, 128, 128]) # 1/4
        # torch.Size([1, 1024, 64, 64]) # 1/8
        # torch.Size([1, 1024, 32, 32]) # 1/16
        # torch.Size([1, 1024, 16, 16]) # 1/32
        # torch.Size([1, 1, 512, 512]) # 1/1 (B, C, 512, 512)

        return self.net(image)[-1] # only last map (== output)

    def _get_network(self):
        net = get_network(
            network_name=self.opt.MODEL.NETWORK_NAME,
            in_chans=self.opt.MODEL.IN_CHANNELS,
            num_classes=self.opt.MODEL.NUM_CLASSES
        )
        net.apply(init_weights)

        if self.opt.MODEL.LOAD_PATH:
            checkpoint = torch.load(self.opt.MODEL.LOAD_PATH, map_location='cpu')
            net.load_state_dict(checkpoint['state_dict'])
            if self.rank == 0:
                print('Loading checkpoint.....')
           
        elif self.opt.MODEL.PRETRAINED_PATH:
            if self.opt.MODEL.IS_RESUME:
                checkpoint = torch.load(self.opt.MODEL.PRETRAINED_PATH)
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net = load_pretrained_weight(net, self.opt.MODEL.PRETRAINED_PATH)
        
        return self._wrap_ddp(net)


    def get_loss(self, output, label):
        return self.loss_fn(output, label)

    # def predict(self, output, label)