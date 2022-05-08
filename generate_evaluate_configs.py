#%%
import numpy as np
import os 
import argparse 
import json 
from imagenet_labels import * 
from itertools import product 

#%%

# def main():

attack_config_path = 'configs/attack'
save_path = 'configs/eval'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for c in os.listdir(attack_config_path):
    tmp_path = os.path.join(attack_config_path, c)

    with open(tmp_path, "r") as jn:
        config = json.load(jn)

    tmp_config = config.copy()

    tmp_config['opt_path'] = 'eval_models/msvd_'
    tmp_config['eval_model_path'] = 'eval_models'
    tmp_config['results_dir'] = 'results'
    
    for em in ['resnet152', 'vgg16']:
        tmp_config['opt_path'] = f'eval_models/msvd_{em}/opt_info.json'
        tmp_config['eval_cnn'] = em 
        
        save_to = os.path.join(save_path, em + '_'+ c)

        with open(save_to, 'w') as writer:
            json.dump(tmp_config, writer)

# %%
