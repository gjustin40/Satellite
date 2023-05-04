import yaml
from yaml.loader import SafeLoader
from easydict import EasyDict

with open('config.yaml') as f:
    opt = yaml.load(f, SafeLoader)
    opt = EasyDict(opt)


opt.SAVE_DIR = 'AAA'
print('aaaaaaaaaaa')

opt = dict(opt)
print(type(opt))
with open('./config_save.yaml', 'w') as f:
    yaml.dump(opt, f, sort_keys=False, default_flow_style=False)