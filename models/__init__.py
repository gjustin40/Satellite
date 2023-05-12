import importlib
from .networks import *
from .base import BaseModel

def find_model_using_name(model_name):
    """"Import the module 'models/[model_name]_model.py'
    """

    model_filename = 'models.' + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    target_model_name = model_name.replace('_', '') + 'Model'
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def get_model(opt):
    return find_model_using_name(opt.MODEL.MODEL_NAME)(opt)
    
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