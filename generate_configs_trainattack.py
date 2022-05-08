
#%%

import numpy as np
import os 
import argparse 
import json 
from imagenet_labels import * 
from itertools import product 
from TREMBA.train_generator import *
#%%

# def main():

# Load initial training configuration file 
parent_config_path = "train_config.json"
with open(parent_config_path, "r") as jn:
    config = json.load(jn)

attack_target_classes = ["English springer, English springer spaniel", "chain saw, chainsaw", "French horn, horn", "parachute, chute"]

# Saving the configurations to load
reverse_labels = dict(zip(imagenet_labels.values(), imagenet_labels.keys()))
config_path = "configs/generator"
for t in attack_target_classes:
    train_config = config.copy()
    target = reverse_labels[t]

    train_config['target'] = target 
    # If target is given, change save path
    if train_config['target'] is not None:
        save_name = "Imagenet_{}_target_{}.pytorch".format("_".join(train_config['model_list']), target)
    else:
        save_name = "Imagenet_{}_untarget.pytorch".format("_".join(train_config['model_list']))

    train_config['save_name'] = save_name 

    config_save_name = save_name.split(".")[0] + '.json'
    if not os.path.exists(config_path):
        os.makedirs(config_path)

    config_save_path = os.path.join(config_path, config_save_name)
    with open(config_save_path, 'w') as writer:
        json.dump(train_config, writer)
#%%
# save_name = f"Imagenet_{}_target_{}"



# %%
