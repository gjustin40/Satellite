import yaml
from yaml.loader import SafeLoader
from easydict import EasyDict

with open('configs/config.yaml') as f:
    opt = yaml.load(f, SafeLoader)

with open('./config_save.yaml', 'w') as f:
    yaml.dump(opt, f, sort_keys=False)

