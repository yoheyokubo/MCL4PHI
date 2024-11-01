import glob
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
paths = glob.glob('./model_save_path_phi_v2/*.csv')
dfs = []
for path in paths:
  dfs.append(pd.read_csv(path, index_col=0).T)
df = pd.concat(dfs, ignore_index=True)
ranks = ['species', 'genus', 'family']
#rows_dp =  ['dp_' + rank for rank in ranks]
#rows_map = ['map_' + rank + '_KNN_1' for rank in ranks]
df = df.rename(columns=lambda x: 'tst_' + x if x not in ['strategy_emd', 'seed'] else x)
#df['strategy_perp'] = '5.0'
#df['strategy_gamm'] = '0.01'
#df['strategy_hrms_alph'] = '2'
#df['strategy_hrms_beta'] = '2'
#df['strategy_hrms_gamm'] = '1.0'
import matplotlib.pyplot as plt
import seaborn as sns

def process_barplot(df, rows):
  df = df.copy()
  df = df.astype({row: 'float32' for row in rows})
  df = df[df['strategy_emd'].isin(['hrms','tsne','HMI','HMIL', 'HMI+KNN1', 'HMIL+KNN1'])]
  df['config.'] = df.apply(lambda x: (x['strategy_emd']), axis=1)
  df = df.sort_values(['config.'])
  fig, axes = plt.subplots(2, 3, tight_layout=True, figsize=(15,15))
  axes = axes.reshape(-1)
  map_max = [0.7142857142857143, 0.7575757575757576, 0.75, 0.8571428571428571, 0.9, 1.0]
  for i, ax in enumerate(axes):
    ax = sns.pointplot(data=df, x="config.", y=rows[i], hue='strategy_emd', errorbar=("sd", 2), ax=ax, linestyle="none", legend=False)
    ax = sns.stripplot(data=df, x="config.", y=rows[i], ax=ax, s=2, color='black')
    ax.set_title(rows[i])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
    if 'map' in rows[i]:
      ax.axhline(y=map_max[i], c="red", ls='--', linewidth=0.8)
  plt.show()
