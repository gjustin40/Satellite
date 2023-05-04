import yaml
from yaml.loader import SafeLoader
from easydict import EasyDict

with open('config.yaml') as f:
    opt = yaml.load(f, SafeLoader)
    opt = EasyDict(opt)

print(opt)

opt.SAVE_DIR = 'AAA'
print('aaaaaaaaaaa')
print(opt)
with open('./config_save.yaml', 'w') as f:
    yaml.dump(dict(opt), f, sort_keys=False)