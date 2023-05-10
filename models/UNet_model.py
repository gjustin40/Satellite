
from .base import BaseModel
from .networks import *
from utils import *

class UNetModel(BaseModel):
    def __init__(self, opt):
        super(UNetModel, self).__init__(opt=opt)
        self.net = self._get_network()

        # Loss Functions
        self.loss_fn = BCELoss()

        # Metric Functions
        # self.metric_fn = Dice

    def forward(self, image):
        return self.net(image)

    def get_loss(self, output, label):
        return self.loss_fn(output, label)

    # def get_metric(self, output, label):
    #     pred = (torch.sigmoid(output.cpu()) > self.opt.CHECKPOINT.THRESHOLD).float()
    #     dice_score = self.metric_fn(pred, label.cpu())
    #     return dice_score
    # def predict(self, output, label)