from .resnet import ResNet

def create_model(opt, rank):
    if opt.MODEL_NAME in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        model = ResNet(opt, rank)

    # if opt.MODEL_NAME in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
    #     model = VGG(opt)
    
    return model