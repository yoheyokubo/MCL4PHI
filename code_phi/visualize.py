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
    #plot_stat = {'hue': hue_name, 'style': style_name, 'markers': markers, 'hue_order': hue_order, 'legend': legend}
    ax = sns.scatterplot(df, x='tSNE 1', y='tSNE 2', ax=ax, hue='names_superkingdom', s=5, style='status', style_order=['seen', 'unseen'])
    
    if names is not None:
        names_array = df['names'].to_numpy()
        u, indices = np.unique(names_array, return_index=True)
        texts = [ax.text(df.at[ind, 'tSNE 1'], df.at[ind, 'tSNE 2'], '{}'.format(names_array[ind]), ha='center', va='center', fontsize='x-small') for ind in names]
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='purple', lw=0.3))

    if show_ids:
        texts = [ax.text(df.at[ind, 'tSNE 1'], df.at[ind, 'tSNE 2'], '{}'.format(ind), ha='center', va='center', fontsize='x-small') for ind in range(data.shape[0])]
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='purple', lw=0.3), time_lim=5, expand=(1.5, 2.0))

    if interactions_list is not None:
        x = df['tSNE 1'].to_numpy()
        y = df['tSNE 2'].to_numpy()
        t = np.linspace(0, 1, 50).reshape((-1,1))
        interactions_trn = np.concatenate([interactions_list[0], interactions_list[1]])
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

    parser.add_argument('--embed_path',  default='', type=str, required=False, help='the path of a pretrained model')
    parser.add_argument('--fsize',  default=10, type=int, required=False, help='the path of a pretrained model')
    parser.add_argument('--show_names', action='store_true', help="whether or not show names of groups when visualizing")
    parser.add_argument('--show_interact', action='store_true', help="whether or not show interactions when visualizing")
    parser.add_argument('--show_ids', action='store_true', help="whether or not show ids of groups when visualizing")
    parser.add_argument('--species',  default='', type=str, required=False, help='the scientific name of the species you are intrested in')
    parser.add_argument('--dist_type',  default='fcgr', type=str, required=False, help='type of distance when calculating pairwise distance of species')
    parser.add_argument('--gamma_p',  default=1.0, type=float, required=False, help='bandwidth hyperparameter (gamma) for phage when using gip for dist_type')
    parser.add_argument('--gamma_h',  default=1.0, type=float, required=False, help='bandwidth hyperparameter (gamma) for host  when using gip for dist_type')
    args = parser.parse_args()

    with open('data_phi/species_fcgr_id_k6_m1.0.json') as f:
        taxids = json.load(f)
    with open('data_phi/scientific_names.json') as f:
        scientific_names = json.load(f)
    with open('data_phi/lineages.json') as f:
        lineages = json.load(f)
    with open('data_phi/phage_host_interaction_pos_neg_list.json') as f:
        interactions = json.load(f)
   
    # convert lineages into a valid format (e.g., None -> -1), but we do not accept such type as [None,...,None]
    lineages = {k: [taxid if isinstance(taxid, int) else -1 for taxid in v] for k,v in lineages.items()}
    hue_order = set()
    for lin in lineages.values():
        hue_order = hue_order | set(map(str, lin))
    hue_order = ['-1'] + list(hue_order - {'-1'})
    lineages = np.array([lineages[tid] for tid in taxids], dtype=np.int32)

    # define the local order
    valid_set = len(interactions[0])
    test_set = valid_set + len(interactions[1])
    interactions = np.concatenate(interactions)
    taxid2locid_dic = {tid: i for i, tid in enumerate(taxids)}
    data = np.array([[taxid2locid_dic[tid_hst], taxid2locid_dic[tid_phg], label] for tid_hst, tid_phg, label in interactions], dtype=np.int32)
    interactions_list = [data[:valid_set], data[valid_set:test_set], data[test_set:]]
    interactions_list_pos = [data[data[:,2] == 1] for data in interactions_list]

    if len(args.embed_path) == 0:
        if args.dist_type == 'fcgr':
            imgs = np.load('data_phi/species_fcgr_np_k6_m1.0.npy')
            imgs = imgs.reshape(imgs.shape[0], -1)
            dist = np.zeros((imgs.shape[0], imgs.shape[0]), dtype=np.float32)
            batch_size = 1000
            for i in range(0, imgs.shape[0], batch_size):
                end_i = min(imgs.shape[0], i+batch_size)
                for j in range(0, imgs.shape[0], batch_size):
                    end_j = min(imgs.shape[0], j+batch_size)
                    dist_local = np.sqrt(np.sum((imgs[i:end_i,None] - imgs[None,j:end_j]) ** 2, axis=2))
                    dist[i:end_i, j:end_j] = dist_local
            figpath_base = 'model_save_path_phi_v5/dist_fcgr.png'
        elif args.dist_type == 'gip':
            dist = np.zeros((len(taxids), len(taxids)), dtype=np.float32)
            arange_hst = np.arange(len(taxids))[lineages[:,6] != 10239]
            arange_phg = np.arange(len(taxids))[lineages[:,6] == 10239]
            interactions_all = np.concatenate(interactions_list_pos)
            for id_hst, id_phg, _ in interactions_all:
                dist[id_hst, id_phg] = 1
                dist[id_phg, id_hst] = 1
            # gaussian interaction profile
            batch_size = 1000
            normalize = np.sum(dist[np.ix_(arange_hst, arange_phg)] ** 2)
            # for host
            adj_hp = dist[np.ix_(arange_hst, arange_phg)]
            normalize_hst = normalize / len(arange_hst)
            #print(dist.shape)
            #print(arange_hst.shape)
            for i in range(0, len(arange_hst)):
                dist[arange_hst[i], arange_hst] = np.sum((adj_hp[i,None]-adj_hp)**2, axis=-1)
            dist[np.ix_(arange_hst, arange_hst)] = np.exp(-args.gamma_h * dist[np.ix_(arange_hst, arange_hst)] / normalize_hst)
            # for phage
            adj_ph = dist[np.ix_(arange_phg, arange_hst)]
            normalize_phg = normalize / len(arange_phg)
            for i in range(0, len(arange_phg), batch_size):
                end_i = min(len(arange_phg), i + batch_size)
                for j in range(0, len(arange_phg), batch_size):
                    end_j = min(len(arange_phg), j + batch_size)
                    dist[np.ix_(arange_phg[i:end_i], arange_phg[j:end_j])] = np.sum((adj_ph[i:end_i,None]-adj_ph[None,j:end_j])**2, axis=-1)
            dist[np.ix_(arange_phg, arange_phg)] = np.exp(-args.gamma_p * dist[np.ix_(arange_phg, arange_phg)] / normalize_phg)
            dist = 1/np.maximum(dist, 1e-6)
            figpath_base = 'model_save_path_phi_v5/dist_gip_p{}_h{}.png'.format(args.gamma_p, args.gamma_h)
        data = TSNE(n_components=2, random_state=42, metric='precomputed', init='random').fit_transform(dist)
    else:
        embed = np.load(args.embed_path)
        embed = embed if len(embed.shape) == 2 else embed[:,0]
        data = TSNE(n_components=2, random_state=42).fit_transform(embed)
        figpath_base = args.embed_path
    df_tsne = pd.DataFrame(data, columns=['tSNE 1', 'tSNE 2'])

    df = pd.DataFrame({'taxid': taxids, 
                       'taxid_genus': [str(tid) for tid in lineages[:,1]],
                       'names': [scientific_names[tid] for tid in taxids],
                       'names_genus': [scientific_names[str(tid)] \
                           if str(tid) in scientific_names else 'no info' \
                           for tid in lineages[:,1]],
                       'names_superkingdom': [scientific_names[str(tid)] \
                           if str(tid) in scientific_names else 'no info' \
                           for tid in lineages[:,6]],
                       'status': ['unseen' for tid in taxids]})
    interactions_trn = np.concatenate([interactions_list_pos[0], interactions_list_pos[1]])
    for id_hst, id_phg, _ in interactions_trn:
        df.loc[id_hst, 'status'] = 'seen'
        df.loc[id_phg, 'status'] = 'seen'
    df = pd.concat([df, df_tsne], axis=1)

    dir_path = os.path.join(os.path.dirname(figpath_base), 'fig')
    os.makedirs(dir_path, exist_ok=True)

    if len(args.species) > 0:
        names2tid = {v:k for k,v in scientific_names.items()}
        tid = names2tid[args.species]
        locid = taxid2locid_dic[tid]
        interactions = np.concatenate([interactions_list_pos[0], interactions_list_pos[1], interactions_list_pos[2]])
        species_sp = scientific_names[str(lineages[locid,6])]
        species_sp_loc = 0 if species_sp in ['Bacteria', 'Archaea'] else 1
        locs_infect = interactions[interactions[:,species_sp_loc] == locid][:,1-species_sp_loc]
        print('infectable species with '+args.species+':')
        for loc_infect in locs_infect:
            x = df.at[loc_infect, 'tSNE 1']
            y = df.at[loc_infect, 'tSNE 2']
            print(scientific_names[taxids[loc_infect]], x, y)
        names_list = [locid] + list(locs_infect)

    if args.show_interact:
        fig_path = os.path.join(dir_path, os.path.splitext(os.path.basename(figpath_base))[0] + '-interact.jpg')
        draw_phi(fig_path, df, interactions_list=interactions_list_pos)
    if args.show_names:
        fig_path = os.path.join(dir_path, os.path.splitext(os.path.basename(figpath_base))[0] + args.species + '-names.jpg')
        if len(args.species) == 0:
            names = np.arange(len(taxids))[lineages[:,6] != 10239].tolist() # host
        else:
            names = names_list 
        draw_phi(fig_path, df, names=names)
    if args.show_ids:
        fig_path = os.path.join(dir_path, os.path.splitext(os.path.basename(figpath_base))[0] + '-ids.jpg')
        draw_phi(fig_path, df, show_ids=True)
    print('finished')
