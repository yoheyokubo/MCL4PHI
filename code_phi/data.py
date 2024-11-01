# operations common among all methods

import os
import numpy as np
import torch
import pickle
import json
import random
from collections import defaultdict
from torch.utils.data import Dataset, Sampler, DataLoader
from torch_geometric.loader import LinkNeighborLoader

# basic (meta) data
class Data:
    def __init__(self, data_dir, granularity, global_ids, data_type='cherry'):
        # load meta data
        data_dir = os.path.join(data_dir, '')
        with open(data_dir + 'scientific_names.json', 'r') as f1,\
             open(data_dir + 'lineages.json', 'r') as f2, \
             open(data_dir + 'phage_host_interaction_pos_neg_list.json', 'r') as f3, \
             open(data_dir + 'phage_host_accs.json') as f4:
            self.scientific_names = json.load(f1)
            lineages = json.load(f2)
            interactions = json.load(f3)
            phage_host_accs = json.load(f4)
        
        # convert lineages into a valid format (e.g., None -> -1), note we do not accept such type as [None,...,None] (where all are None)
        lineages = {k: [taxid if isinstance(taxid, int) else -1 for taxid in v] for k,v in lineages.items()}
        if granularity == 'contigs':
            temp = {}
            for k in lineages.keys():
                if k in phage_host_accs:
                    for acc in phage_host_accs[k]:
                        temp[acc] = lineages[k]
            lineages = temp
        self.lineages = np.array([lineages[gid] for gid in global_ids], dtype=np.int32)

        # define the global order over the whole dataset at species level
        #taxids = np.unique(np.array(interactions)[:,0:2])
        #taxid2locid_dic = {tid: i for i, tid in enumerate(taxids)}
        #interactions_list = [np.array([[taxid2locid_dic[tid_hst], taxid2locid_dic[tid_phg], label] \
        #        for tid_hst, tid_phg, label in interactions], dtype=np.int32) \
        #        for interaction in interactions]
        #self.lineages = torch.tensor([lineages[tid] for tid in taxids], dtype=torch.int32)
        #accs = []
        #for tid in taxids:
        #    for acc in phage_host_accs[tid]:
        #        accs.append(acc)
        #acc2locid_dic = {acc: i for i, acc in enumerate(accs)}
        #self.global_ids_dic = {'species': np.array(taxids), 'contigs': np.array(accs)}


        interactions_list = [np.array(interaction) for interaction in interactions]
        if data_type == 'cherry':
            self.dataset_stat = [
                    {'name': 'trn', 'query': 'none'},
                    {'name': 'val', 'query': 'none'},
                    {'name': 'tst', 'query': 'phage'}]
        elif data_type == 'phd':
            interactions_list = self.data_split_phd(interactions_list)
            self.dataset_stat = [
                    {'name': 'trn', 'query': 'none'},
                    {'name': 'val', 'query': 'none'},
                    {'name': 'tst_phg_unseen', 'query': 'phage'},
                    {'name': 'tst_phg_unseen_reduced', 'query': 'phage'},
                    {'name': 'tst_host_unseen', 'query': 'host'},
                    {'name': 'tst_both_unseen', 'query': 'pair'}]



        # convert taxids (or accession number) to location index in each dataset
        mapper = {gid: i for i, gid in enumerate(global_ids)}
        if granularity == 'species':
            self.interactions_list = [np.array([[mapper[tid_hst], mapper[tid_phg], int(label)] \
                    for tid_hst, tid_phg, label in interactions if tid_hst in mapper and tid_phg in mapper], dtype=np.int32) \
                    for interactions in interactions_list]
        elif granularity == 'contigs': # expand species-level interaction information to contig-level one
            self.interactions_list = []
            for interactions in interactions_list:
                temp = []
                for tid_hst, tid_phg, label in interactions:
                    for acc_hst in phage_host_accs[tid_hst]:
                        for acc_phg in phage_host_accs[tid_phg]:
                            if acc_hst in mapper and acc_phg in mapper:
                                temp.append([mapper[acc_hst], mapper[acc_phg], int(label)])
                self.interactions_list.append(np.array(temp, dtype=np.int32))
        
        # identify training species (or contigs) for later normalization
        self.trn_ids_dic = defaultdict(bool)
        trn_ids = np.unique(self.interactions_list[0][0:2])
        for trn_id in trn_ids:
            self.trn_ids_dic[trn_id] = True

    def data_split_phd(self, interactions_list):
        # further split test dataset into phage-unseen, phage-unseen-reduced, host-unseen, both-unseen
        interactions_seen = np.concatenate(interactions_list[:2])
        hst_seen = np.unique(interactions_seen[:,0]) # seen in either positive or negative interaction
        phg_seen = np.unique(interactions_seen[:,1])
        interactions_tst = interactions_list[2]
        mask_hst_seen = np.isin(interactions_tst[:,0], hst_seen)
        mask_phg_seen = np.isin(interactions_tst[:,1], phg_seen)
        interactions_phg_unseen  = interactions_tst[mask_hst_seen & (~mask_phg_seen)]
        interactions_hst_unseen  = interactions_tst[(~mask_hst_seen) & mask_phg_seen]
        interactions_both_unseen = interactions_tst[(~mask_hst_seen) & (~mask_phg_seen)]
        # phage-unseen-reduced is created for comparison with some of the existings models
        phg_unseen = np.unique(interactions_phg_unseen[:,1])
        phg_reduced = []
        for tid_phg in phg_unseen:
            label_satus = set(interactions_phg_unseen[interactions_phg_unseen[:,1] == tid_phg][:,2])
            if '1' not in label_satus: # if there is no positive interaction with this phage
                phg_reduced.append(tid_phg)
        phg_reduced = np.array(phg_reduced)
        mask_reduced = np.isin(interactions_phg_unseen[:,1], phg_reduced)
        interactions_phg_unseen_reduced = interactions_phg_unseen[~mask_reduced] # filtered out
        interactions_list = [interactions_list[0], interactions_list[1],
                             interactions_phg_unseen, interactions_phg_unseen_reduced,
                             interactions_hst_unseen, interactions_both_unseen]
        return interactions_list

    def set_scaler(self, X_trn, norm_type='standard', local_norm=False):
        if local_norm:
            axis = 0 # norm is conducted over all training samples (species) for each element (e.g., each kmer term)
        else:
            axis = None
        if norm_type == 'none':
            self.mn = 0
            self.sc = 1
        elif norm_type == 'minmax':
            self.mn = X_trn.min(axis=axis, keepdims=True)
            self.sc = X_trn.max(axis=axis, keepdims=True) - self.mn
        elif norm_type == 'standard':
            self.mn = X_trn.mean(axis=axis, keepdims=True)
            self.sc = X_trn.std(axis=axis, keepdims=True)
    
    def norm(self, X):
        return (X - self.mn) / self.sc

    def renorm(self, X):
        return X * self.sc + self.mn

# datast of positive and negative interactions to construct loss function
class InteractionDataset(Dataset):
    def __init__(self, interactions):
        self.interactions = interactions
        print(interactions.shape)

    def __len__(self):
        return self.interactions.shape[0]

    def __getitem__(self, idx):
        return self.interactions[idx]

    def my_sampler(self, **kwargs):
        return MySubsetRandomSampler(self.interactions, **kwargs)

# sample a subset of all postive and negative interactions (ratios change according to training or testing modes)
class MySubsetRandomSampler(Sampler):
    def __init__(self, interactions, pos_ratio=None, neg_ratio=None, pos_neg_ratio=None, pos_samples=None, neg_samples=None):
        indices = np.arange(interactions.shape[0])
        self.indices_pos = indices[interactions[:,2] == 1].tolist()
        self.indices_neg = indices[interactions[:,2] == 0].tolist()
        assert pos_ratio is not None or pos_samples is not None
        self.samples_pos = int(len(self.indices_pos) * pos_ratio)\
                if pos_ratio is not None else pos_samples
        assert neg_ratio is not None or pos_neg_ratio is not None or neg_samples is not None
        self.samples_neg = int(len(self.indices_neg) * neg_ratio)\
                if neg_ratio is not None else int(self.samples_pos * pos_neg_ratio)\
                if (pos_neg_ratio is not None and pos_neg_ratio >= 0) else len(self.indices_neg)\
                if (pos_neg_ratio is not None and pos_neg_ratio < 0) else neg_samples

    def __len__(self):
        return self.samples_pos + self.samples_neg
  
    def __iter__(self):
        l = (random.sample(self.indices_pos, self.samples_pos) if len(self.indices_pos) > self.samples_pos else self.indices_pos)\
            + (random.sample(self.indices_neg, self.samples_neg) if len(self.indices_neg) > self.samples_neg else self.indices_neg)
        random.shuffle(l)
        yield from l


def get_dataloaders(data, pos_neg_ratios=[1.0,-1,-1,-1,-1,-1], batch_size=32, num_workers=4):
    # interaction dataset to construct loss
    datasets = [data.get_dataset(i) for i in range(len(pos_neg_ratios))] 
    my_collate_fn = datasets[0].my_collate_fn
    dataloaders = [DataLoader(datasets[i], batch_size=batch_size, collate_fn=datasets[i].my_collate_fn, \
                    sampler=datasets[i].my_sampler(pos_ratio=1.0, pos_neg_ratio=pos_neg_ratios[i]), num_workers=num_workers)\
                    for i in range(len(pos_neg_ratios))]
    return [{'loader': dataloaders[i], 'data': data} for i in range(len(pos_neg_ratios))]

def get_graphloaders(data, pos_neg_ratios=[-1]*6, graph_type='hetero', batch_size=64, num_workers=4):
    datasets = [data.get_dataset(i) for i in range(len(pos_neg_ratios))]
    if graph_type == 'hetero':
        dataloaders = [LinkNeighborLoader(
            datasets[i].graph_data, # only the attribute 'edge_index' (or 'adj' if any) is used for samling from seed edges
            num_neighbors=[5,5],
            batch_size=batch_size,
            edge_label_index=(('phage', 'infects', 'host'), datasets[i].graph_data['phage', 'infects', 'host'].edge_label_index), # seed edges
            edge_label=datasets[i].graph_data['phage', 'infects', 'host'].edge_label,
            num_workers=num_workers,
            sampler=datasets[i].my_sampler(pos_ratio=1.0, pos_neg_ratio=pos_neg_ratios[i]) # we don't use shuffle=True because our sampler will do this
        ) for i in range(len(pos_neg_ratios))] # maximum number of samples = 2 (= phages and hosts) * 5 * 5 (= 2 hops) * 64 (= batch_size for seed edge) per batch
    elif graph_type == 'homo':
        dataloaders = [LinkNeighborLoader(
            datasets[i].graph_data, # only the attribute 'edge_index' (or 'adj' if any) is used for samling from seed edges
            num_neighbors=[5,5],
            batch_size=batch_size,
            edge_label_index=datasets[i].graph_data.edge_label_index, # seed edges
            edge_label=datasets[i].graph_data.edge_label,
            num_workers=num_workers,
            sampler=datasets[i].my_sampler(pos_ratio=1.0, pos_neg_ratio=pos_neg_ratios[i]) # we don't use shuffle=True because our sampler will do this
        ) for i in range(len(pos_neg_ratios))] # maximum number of samples = 2 (= phages and hosts) * 5 * 5 (= 2 hops) * 64 (= batch_size for seed edge) per batch
    
    return [{'loader': dataloaders[i], 'dataset': datasets[i], 'data': data} for i in range(len(pos_neg_ratios))]

