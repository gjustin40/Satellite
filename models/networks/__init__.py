from .unet import *

def get_network(network_name, in_channels=3, num_classes=1):
    net = eval(network_name)(
        in_channels=in_channels,
        num_classes=num_classes
    )

    return net