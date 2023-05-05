from .unet import *
from .beit_adapter_upernet import *
from .beit_adapter_upernet_aux import *
from .beit_adapter import *
from .beit import *

def get_network(network_name, in_chans=3, num_classes=1):
    net = eval(network_name)(
        in_chans=in_chans,
        num_classes=num_classes
    )

    return net