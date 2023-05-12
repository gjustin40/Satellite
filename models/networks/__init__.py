from .unet import *
from .beit_adapter_upernet import *
from .beit_adapter_upernet_aux import *
from .beit_adapter_upernet_kd import *
from .beit_adapter import *
from .beit import *
import torch.nn as nn

def get_network(network_name, in_chans=3, num_classes=1):
    # binary segmentation only support sigmoid activation function
    # num_classes == 1 or 2 should be considered with binary
    # num_classes == 1 or 2 should be 1 channel of ouput
    if num_classes == 2:
        num_classes = 1

    net = eval(network_name)(
        in_chans=in_chans,
        num_classes=num_classes
    )

    return net

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out // m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)