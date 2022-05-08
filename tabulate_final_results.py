
#%%

import pandas as pd 
import os 
#%%
root_path = 'results'
results = [os.path.join(root, name)
for root, dirs, files in os.walk(root_path)
for name in files
if name.endswith("scored.csv")]
# %%
dfs = []
for r in results:
    if '0.03125' in r:
        continue 
    df = pd.read_csv(r)
    dfs.append(df)
fulldf = pd.concat(dfs)
# %%
result_df_rows = []
for name, group in fulldf.groupby(['source_model', 'eval_model', 'epsilon']):
    success_rate = group['success'].mean()

    row = [*name, success_rate]
    result_df_rows.append(row)
result_df = pd.DataFrame(result_df_rows, columns=['source_model', 'eval_model', 'epsilon', 'success_rate'])
# %%
result_df.to_csv("results/final_results.csv", index=False)
# %%
