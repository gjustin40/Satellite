import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

configs = {
    'vgg11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    
    def __init__(self, config, in_channels=3, num_classes=10, batch_norm=True):
        super(VGG, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        
        self.features = self.make_features()
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.flatten = nn.Flatten()
        self.classifier = self.make_classifier()
        # self._weight_init()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x
        
    def make_features(self):
        in_channels = self.in_channels
        config = self.config
        batch_norm = self.batch_norm
        layers = []
        for out_channels in configs[config]:
            if type(out_channels) == int:
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)] 
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                    
                in_channels = out_channels
            else:
                layers += [nn.MaxPool2d(2)]
                
        return nn.Sequential(*layers)
    
    def make_classifier(self):
        num_classes = self.num_classes
        
        layers = []
        layers += [nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
        layers += [nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
        layers += [nn.Linear(4096, num_classes)]
        
        return nn.Sequential(*layers)

    def _weight_init(self):
        for m in self.modules():
            if (type(m) == nn.Conv2d) or (type(m) == nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

configs = {
    'vgg11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
weights = {
    'vgg11': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}

def resnet34(num_classes=1000, pretrained=False):
    net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(weights['resnet34']), strict=False)
    net.fc = nn.Linear(net.in_planes, num_classes)
    
    return net


def vgg11(num_classes=1000, pretrained=False):
    net = VGG('vgg11', num_classes=1000)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(weights['vgg11']))
    net.num_classes = num_classes
    net.classifier = net.make_classifier()

    return net

def vgg13(num_classes=1000, pretrained=False):
    net = VGG('vgg13', num_classes=1000)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(weights['vgg13']))
    net.num_classes = num_classes
    net.classifier = net.make_classifier()

    return net

def vgg16(num_classes=1000, pretrained=False):
    net = VGG('vgg16', num_classes=1000)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(weights['vgg16']))
    net.num_classes = num_classes
    net.classifier = net.make_classifier()

    return net

def vgg19(num_classes=1000, pretrained=False):
    net = VGG('vgg19', num_classes=1000)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(weights['vgg19']))
    net.num_classes = num_classes
    net.classifier = net.make_classifier()

    return net

if __name__ == '__main__':
    model = VGG('vgg11', num_classes=2)
    print(model)