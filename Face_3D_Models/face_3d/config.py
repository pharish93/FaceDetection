import numpy as np
from easydict import EasyDict as edict

config = edict()

# default settings
default = edict()

# default network
default.network = 'vgg'
default.pretrained = 'model/vgg16'
default.pretrained_epoch = 0
default.base_lr = 0.001

# default dataset
default.dataset = 'AFLW'


#Harish modified

default.root_path = '/home/harish/data/'
# default.root_path = '/home/ubuntu/data/'
# default.root_path = '/data/datasets/AFLW/'
default.dataset_path = 'aflw/data'

# network settings
network = edict()
network.vgg = edict()

# dataset settings
dataset = edict()
dataset.AFLW = edict()



def generate_config(_network, _dataset):
    for k, v in network[_network].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
    for k, v in dataset[_dataset].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v

