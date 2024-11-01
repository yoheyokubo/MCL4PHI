# train the constrastive learning
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from adjustText import adjust_text
from collections import defaultdict
import json
import os

def tSNE_fit(X, df):
    data = TSNE(n_components=2, random_state=42).fit_transform(X)
    df2 = pd.DataFrame(data, columns=['tSNE 1', 'tSNE 2'])
    df = pd.concat([df, df2], axis=1)
    return df

def draw_phi(fig_path, df, names=None, interactions_list=None, fsize=15, show_ids=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=(fsize,fsize), dpi=300)
    ax = axes[0,0]
    ax = sns.scatterplot(df, x='tSNE 1', y='tSNE 2', ax=ax, hue='names_superkingdom', s=15, style='status', style_order=['seen', 'unseen'])
    
    if names is not None:
        names_array = df['names'].to_numpy()
        u, indices = np.unique(names_array, return_index=True)
        texts = [ax.text(df.at[ind, 'tSNE 1'], df.at[ind, 'tSNE 2'], '{}'.format(names_array[ind]), ha='center', va='center', fontsize='x-small') for ind in names]
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='purple', lw=0.3), time_lim=5, expand=(1.5, 2.0))

    if show_ids:
        texts = [ax.text(df.at[ind, 'tSNE 1'], df.at[ind, 'tSNE 2'], '{}'.format(ind), ha='center', va='center', fontsize='x-small') for ind in range(data.shape[0])]
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='purple', lw=0.3), time_lim=5, expand=(1.5, 2.0))

    if interactions_list is not None:
        x = df['tSNE 1'].to_numpy()
        y = df['tSNE 2'].to_numpy()
        t = np.linspace(0, 1, 50).reshape((-1,1))
        interactions_trn = interactions_list[0] #np.concatenate([interactions_list[0], interactions_list[1]])
        x1 = x[interactions_trn[:,0]]
        x2 = x[interactions_trn[:,1]]
        y1 = y[interactions_trn[:,0]]
        y2 = y[interactions_trn[:,1]]
        x_plot = x1 + (x2-x1) * t
        y_plot = y1 + (y2-y1) * t
        ax.plot(x_plot, y_plot, color='k', linewidth=0.2, alpha=0.5)
        interactions_tst = interactions_list[2]
        x1 = x[interactions_tst[:,0]]
        x2 = x[interactions_tst[:,1]]
        y1 = y[interactions_tst[:,0]]
        y2 = y[interactions_tst[:,1]]
        x_plot = x1 + (x2-x1) * t
        y_plot = y1 + (y2-y1) * t
        ax.plot(x_plot, y_plot, color='r', linewidth=0.2, alpha=0.5)

    plt.savefig(fig_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='<Representation learning for the whole genome taxonomic classification>')

    show_ids = False
    show_names = True
    show_interact = True
    fsize = 10

    data_dir = '../data_cherry/'

    embed_path = '../data_cherry/cl4phi/fcgr_k6_d0.0_species.npz'
    node_feature = np.load(embed_path)
    embeds = node_feature['x']
    embeds = embeds.reshape((len(embeds), -1))
    global_ids = node_feature['name']
    with open(data_dir + 'scientific_names.json') as f:
        scientific_names = json.load(f)
    with open(data_dir + 'lineages.json') as f:
        lineages = json.load(f)
    with open(data_dir + 'phage_host_interaction_pos_neg_list.json') as f:
        interactions = json.load(f)
   
    # convert lineages into a valid format (e.g., None -> -1), but we do not accept such type as [None,...,None]
    lineages = {k: [taxid if isinstance(taxid, int) else -1 for taxid in v] for k,v in lineages.items()}
    hue_order = set()
    for lin in lineages.values():
        hue_order = hue_order | set(map(str, lin))
    hue_order = ['-1'] + list(hue_order - {'-1'})
    lineages = np.array([lineages[tid] for tid in global_ids], dtype=np.int32)

    # define the local order
    mapper = {tid: i for i, tid in enumerate(global_ids)}
    interactions_list = [np.array([[mapper[tid_hst], mapper[tid_phg], label] for tid_hst, tid_phg, label in interaction], dtype=np.int32)\
                         for interaction in interactions]
    interactions_list_pos = [inc[inc[:,2] == 1] for inc in interactions_list]

    data = TSNE(n_components=2, random_state=42).fit_transform(embeds)
    figpath_base = embed_path
    df_tsne = pd.DataFrame(data, columns=['tSNE 1', 'tSNE 2'])

    df = pd.DataFrame({'gid': global_ids, 
                       'taxid_genus': [str(tid) for tid in lineages[:,1]],
                       'names': [scientific_names[tid] for tid in global_ids],
                       'names_genus': [scientific_names[str(tid)] \
                           if str(tid) in scientific_names else 'no info' \
                           for tid in lineages[:,1]],
                       'names_superkingdom': ['Bacteria' \
                           if tid == 2 else 'Archaea' if tid == 2157 \
                           else 'Virus' if tid == 10239 else 'no info'\
                           for tid in lineages[:,6]],
                       'status': ['unseen' for tid in global_ids]})
    interactions_trn = interactions_list_pos[0] #np.concatenate([interactions_list_pos[0], interactions_list_pos[1]])
    for id_hst, id_phg, _ in interactions_trn:
        df.loc[id_hst, 'status'] = 'seen'
        df.loc[id_phg, 'status'] = 'seen'
    df = pd.concat([df, df_tsne], axis=1)

    dir_path = os.path.join(os.path.dirname(figpath_base), 'fig')
    os.makedirs(dir_path, exist_ok=True)

    if show_interact:
        fig_path = os.path.join(dir_path, os.path.splitext(os.path.basename(figpath_base))[0] + '-interact.jpg')
        draw_phi(fig_path, df, interactions_list=interactions_list_pos, fsize=fsize)
    if show_names:
        fig_path = os.path.join(dir_path, os.path.splitext(os.path.basename(figpath_base))[0] + '' + '-names.jpg')
        names = np.arange(len(global_ids))[lineages[:,6] != 10239].tolist() # host
        draw_phi(fig_path, df, names=names, fsize=fsize)
    if show_ids:
        fig_path = os.path.join(dir_path, os.path.splitext(os.path.basename(figpath_base))[0] + '-ids.jpg')
        draw_phi(fig_path, df, show_ids=True)
    print('finished')
