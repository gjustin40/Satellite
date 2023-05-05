# from models.beit_adapter_upernet_aux import BEiTAdapterUperNetAux
# from models.networks.beit_adapter_upernet import BEiTAdapterUperNet
from models.networks import get_network
import torch
import time
torch.cuda.set_device('cuda:1')
net = get_network(
    network_name='BEiTAdapterUperNet',
    # network_name='UNet',
    in_chans=4,
    num_classes=1
).to('cuda:1')

net.eval()
data = torch.Tensor(1,4,512,512).to('cuda:1')
print(data.shape)
with torch.no_grad():
    net(data)
    out = net(data)

    # print(len(out))
    # for o in out:
    #     print(o.shape)
