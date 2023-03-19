import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.model_zoo as model_zoo

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride):
        super(BasicBlock, self).__init__()        
        
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = self._make_downsample(in_planes, planes, stride)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.downsample(identity)
        out = self.relu(out)
        
        return out
    
    def _make_downsample(self, in_planes, planes, stride):        
        layers = [
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        ]
        
        return nn.Sequential(*layers)
    

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != (planes * self.expansion):
            self.downsample = self._make_downsample(in_planes, planes, stride=stride)
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(identity)
        out = self.relu(out)
        
        return out
    
    def _make_downsample(self, in_planes, planes, stride):
        layers = [
            nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * self.expansion)
        ]
        
        return nn.Sequential(*layers)
        

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_planes, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x
        
    
    def _make_layers(self, block, planes, num_block, stride):
        strides = [stride] + [1]*(num_block - 1)
        layers = []
        for stride in strides:
            layers += [block(self.in_planes, planes, stride=stride)]
            self.in_planes = planes * block.expansion      
        
        return nn.Sequential(*layers)


weights = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth'
}
def resnet18(num_classes=1000, pretrained=False):
    net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(weights['resnet18']))
    net.fc = nn.Linear(net.in_planes, num_classes)
    
    return net

def resnet34(num_classes=1000, pretrained=False):
    net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(weights['resnet34']))
    net.fc = nn.Linear(net.in_planes, num_classes)
    
    return net

def resnet50(num_classes=1000, pretrained=False):
    net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(weights['resnet50']))
    net.fc = nn.Linear(net.in_planes, num_classes)
    
    return net

def resnet101(num_classes=1000, pretrained=False):
    net = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=1000)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(weights['resnet101']))
    net.fc = nn.Linear(net.in_planes, num_classes)
    
    return net

def resnet152(num_classes=1000, pretrained=False):
    net = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=1000)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(weights['resnet152']))
    net.fc = nn.Linear(net.in_planes, num_classes)

    return net


if __name__ == '__main__':
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    weight =  model.load_state_dict(model_zoo.load_url(weights['resnet18']))
