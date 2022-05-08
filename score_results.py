#%%
import pandas as pd
import numpy as np
import time
import os
from TREMBA.dataloader import imagenet
from imagenet_labels import imagenet_labels
from itertools import product 


from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')




#%%

success_threshold = 0.5

################## CONCAT ALL RESULTS ########
root_path = 'results'
results = [os.path.join(root, name)
for root, dirs, files in os.walk(root_path)
for name in files
if name.endswith("eval_summary.csv")]

#%%

#### Determine all combinations of conv and eval models
conv_models = []
eval_models = []
epsilons = []
for r in results:
    if '/' in r:
        base_name = r.split("/")[1]
    else:
        base_name = r.split("\\")[1]
    
    conv_model = base_name.split("_")[0]
    eval_model = base_name.split(".")[1].split("_")[1]
    epsilon = base_name.split("_eval")[0].split("_")[-1]
    
    if conv_model not in conv_models:
        conv_models.append(conv_model)
    
    if eval_model not in eval_models:
        eval_models.append(eval_model)
    
    if epsilon not in epsilons:
        epsilons.append(epsilon)
        
#%%

# Filter by combination and concat the results 
combinations = list(product(conv_models, eval_models, epsilons))

for c in combinations:
    conv_model = c[0]
    eval_model = c[1]
    epsilon = c[2]

    if conv_model == eval_model:
        filter_statement = lambda x: (eval_model in x.replace(conv_model, "", 1)) and (epsilon in x)
    else:
        filter_statement = lambda x: (conv_model in x) and (eval_model in x) and (epsilon in x)

    filt = list(filter(filter_statement, results))
    if len(filt) == 0:
        continue 

    combined_result_path = f"{root_path}/{conv_model}_{eval_model}_{epsilon}results.csv"
    
    dataframes = [pd.read_csv(f) for f in filt]
    concatenated = pd.concat(dataframes)
    concatenated['target'] = concatenated['target'].apply(lambda x: imagenet_labels[x])
    concatenated.to_csv(combined_result_path, index=False)
#%%

######################## LOAD RESULTS AND SCORE SEMANTICALLY #########
# %%

root_path = 'results'
results = [os.path.join(root, name)
for root, dirs, files in os.walk(root_path)
for name in files
if name.endswith("results.csv")]

#%%

label_mapper = {

    "English springer, English springer spaniel": "dog",
    "chain saw, chainsaw": "chainsaw",
    "French horn, horn": "horn",
    "parachute, chute": "parachute"
}

#%%
for r in results:
    df = pd.read_csv(r)
    df['target'] = df['target'].apply(lambda x: label_mapper[x])

    emb1 = model.encode(df['target'], convert_to_tensor=True)
    emb2 = model.encode(df['adversarial_caption_np'], convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(emb1, emb2)
    
    df['score'] = np.diagonal(cos_sim.detach().cpu().numpy())
    df['success'] = df['score'].apply(lambda x: 1 if x >= success_threshold else 0)

    save_path = r[:-4] + "_scored.csv"
    df.to_csv(save_path, index=False)

# %%
