import sys
import argparse
import os
import json
import numpy as np
import time
import pickle
import json
import gc
from collections import defaultdict


base_dir = '../data_cherry/'
pos = int(sys.argv[1]) # assume it ranges from 1 to 100

with open(base_dir + 'phage_accs.json') as f:
    phage_accs = json.load(f)
with open(base_dir + 'host_accs.json') as f:
    host_accs = json.load(f)
phage_dic = defaultdict(bool)
host_dic = defaultdict(bool)
for phage, accs in phage_accs.items():
    phage_dic[phage] = True
    for acc in accs:
        phage_dic[acc] = True
for host, accs in host_accs.items():
    host_dic[host] = True
    for acc in accs:
        host_dic[acc] = True

def get_descriptor(kmer_phg, kmer_hst):
    x = kmer_phg
    y = kmer_hst
    x_mn = np.mean(x, axis=1)
    x_mn_sq = x_mn ** 2
    x_sq_mn = np.mean(x ** 2, axis=1)
    y_mn = np.mean(y, axis=1)
    y_mn_sq = y_mn ** 2
    y_sq_mn = np.mean(y ** 2, axis=1)
    divisor = np.sqrt((x_sq_mn - x_mn_sq)[:,None] * (y_sq_mn - y_mn_sq)[None])
    mask = (divisor > 0).astype(int)
    
    if len(x) * len(y) > 500000: # slow calculation to save memory
        xy_mn = np.zeros((len(x), len(y)))
        n_kmers = x.shape[-1] # 4**kmer = 4096
        bs = max(1, int(2000000000 / (len(x) * len(y))))
        #print(len(x), len(y), bs)
        for i in range(0, n_kmers, bs): # 4**kmer
            end_i = min(n_kmers, i+bs)
            xy_mn += np.sum(x[:,None,i:end_i] * y[None,:,i:end_i], axis=-1)
        xy_mn /= x.shape[-1]
    else: # create (# of local kmers in x, # of local kmers in y, 4**6)
        #print('skipped')
        xy_mn = np.mean(x[:,None] * y[None], axis=-1)
    corr = (xy_mn - x_mn[:,None] * y_mn[None]) / np.maximum(1e-6, divisor) * mask
    corr += (-2) * (1-mask)
    ind = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
    descriptor = (x[ind[0]] - y[ind[1]]).reshape(1,-1)
    return descriptor, np.max(corr, axis=None)

types = ['species', 'contigs']
for t in types:
    row = []
    col = []
    val = []
    corrs = []
    descs = []
    gc.collect()
    with open(base_dir + f'pblks/{t}_dic_pblks_k6.pkl', 'rb') as f:
        dic_kmer = pickle.load(f)
    # define the global order of ids (taxids or accession numbers) over this dataset
    global_ids = list(dic_kmer.keys())
    X_kmer_phg = {gid: np.array(list(dic_kmer[gid].values()), dtype=np.float32) for gid in global_ids if phage_dic[gid]}
    batch_size = int(len(X_kmer_phg)/100) + 1
    i = (pos-1)*batch_size
    end_i = min(len(X_kmer_phg), i+batch_size)
    X_kmer_hst = {gid: np.array(list(dic_kmer[gid].values()), dtype=np.float32) for gid in global_ids if host_dic[gid]}
    #print(set(global_ids) - (set(X_kmer_phg.keys()) | set(X_kmer_hst.keys())))
    #exit(0)
    running_procs = []
    for k1, v1 in list(X_kmer_phg.items())[i:end_i]:
        for k2, v2 in X_kmer_hst.items():
            row.append(k1)
            col.append(k2)
            descriptor, corr = get_descriptor(v1, v2)
            corrs.append(corr)
            descs.append(descriptor)
    edge_index = np.stack([np.array(row), np.array(col)], axis=0)
    corr = np.array(corrs, dtype=np.float32)
    descriptor = np.concatenate(descs)
    np.savez(base_dir + f'pblks/{t}_network_{pos}.npz', edge_index=edge_index, corr=corr, descriptor=descriptor)

print('finished')
