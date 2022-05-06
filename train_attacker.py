
#%%

import numpy as np
import os 
import argparse 
import json 
from imagenet_labels import * 
from itertools import product 
from TREMBA.train_generator import train_tremba
#%%

# def main():

config_path = 'configs/generator'

configs = [f for f in os.listdir(config_path)]
configs
#%%
# save_name = f"Imagenet_{}_target_{}"
for c in configs:
    read_path = os.path.join(config_path, c)
    with open(read_path, 'r') as reader:
        train_config = json.load(reader)


# %%

asdf = train_tremba(train_config)
# %%

#%%
for idx, a in enumerate(asdf):
    print(idx)

    data = a[0]
    labels = a[1]
    print(data)
    print(labels)
    break 
# %%
