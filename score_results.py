#%%
import pandas as pd
import numpy as np
import time
import os
from TREMBA.dataloader import imagenet
from imagenet_labels import imagenet_labels
from itertools import product 
import nltk 
nltk.download('stopwords')

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))


#%%

success_threshold = 0.3

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
for r in results:
    base_name = r.split("\\")[1]
    conv_model = base_name.split("_")[0]
    eval_model = base_name.split(".")[1].split("_")[1]

    if conv_model not in conv_models:
        conv_models.append(conv_model)
    
    if eval_model not in eval_models:
        eval_models.append(eval_model)
#%%

# Filter by combination and concat the results 
combinations = list(product(conv_models, eval_models))

for c in combinations:
    conv_model = c[0]
    eval_model = c[1]

    filter_statement = lambda x: conv_model in x and eval_model in x 

    filt = list(filter(filter_statement, results))
    combined_result_path = f"{root_path}/{conv_model}_{eval_model}_results.csv"
    
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
    df['score'] = cos_sim.detach().cpu().numpy()
    df['success'] = df['score'].apply(lambda x: 1 if x >= success_threshold else 0)

    save_path = r[:-4] + "_scored.csv"
    df.to_csv(save_path, index=False)

# %%
