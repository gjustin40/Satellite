from .resnet import *
from .vgg import *

def build_network(name, num_classes, pretrained=False):
    # ResNet
    if name == 'resnet18':
        return resnet18(num_classes=num_classes, pretrained=pretrained)
    elif name == 'resnet34':
        return resnet34(num_classes=num_classes, pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(num_classes=num_classes, pretrained=pretrained)
    elif name == 'resnet101':
        return resnet101(num_classes=num_classes, pretrained=pretrained)
    elif name == 'resnet152':
        return resnet152(num_classes=num_classes, pretrained=pretrained)

    # VGG
    elif name == 'vgg11':
        return vgg11(num_classes=num_classes, pretrained=pretrained)
    elif name == 'vgg13':
        return vgg13(num_classes=num_classes, pretrained=pretrained)
    elif name == 'vgg16':
        return vgg16(num_classes=num_classes, pretrained=pretrained)
    elif name == 'vgg19':
        return vgg19(num_classes=num_classes, pretrained=pretrained)

if __name__ == '__main__':
    net = get_network('vgg11', num_classes=2, pretrained=True)
    net = get_network('vgg13', num_classes=2, pretrained=True)
    net = get_network('vgg16', num_classes=2, pretrained=True)
    net = get_network('vgg19', num_classes=2, pretrained=True)