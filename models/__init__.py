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
    return find_model_using_name(opt.MODEL_NAME)(opt)
    