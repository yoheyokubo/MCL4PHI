import os
import json
import numpy as np
from pyfaidx import Fasta
from collections import defaultdict
import math
import random
import sys

data_dir = '../data_cherry/'

random.seed(123)
if len(sys.argv) > 0:
    dropout = float(sys.argv[1])
else:
    dropout = 0

with open(data_dir + 'phage_host_accs.json') as f:
    phage_host_accs = json.load(f)


def count_kmers(sequences, k, dropout=0):
    d = defaultdict(int)
    for sequence in sequences:
        for i in range(len(sequence)-(k-1)):
            if dropout == 0 or random.random() > dropout:
                d[sequence[i:i+k]] += 1
        for key in list(d.keys()):
            if "N" in key:
                del d[key]
    return d

kmer = 4
k_list = ["A", "C", "G", "T"]
nucl_list = ["A", "C", "G", "T"]
for i in range(kmer-1):
    tmp = []
    for item in nucl_list:
        for nucl in k_list:
            tmp.append(nucl+item)
    k_list = tmp
# dictionary

wgs = Fasta(data_dir + 'phage_host.fasta')
kmer_species = {}
kmer_contigs = {}
acc_status = defaultdict(bool)
for acc in wgs.keys():
    acc_status[acc] = True
# species level images
for taxid, accs in phage_host_accs.items():
    seqs = [wgs[acc][:].seq.upper() for acc in accs if acc_status[acc]]
    if len(seqs) > 0:
        counts = count_kmers(seqs, kmer, dropout=dropout)
        kmer_species[taxid] = [counts[kmer_term] for kmer_term in k_list]
# contig level images
#for acc in wgs.keys():
#    seqs = [wgs[acc][:].seq.upper()]
#    counts = count_kmers(seqs, kmer)
#    kmer_contigs[acc] = [counts[kmer_term] for kmer_term in k_list]

out_fn = data_dir + 'cherry/'
os.makedirs(os.path.dirname(out_fn), exist_ok=True)
names = np.array(list(kmer_species.keys()))
features = np.array([kmer_species[name] for name in names], dtype=np.float32)
np.savez(out_fn + f'feature_k{kmer}_d{dropout}_species.npz', x=features, name=names)
#names = np.array(list(kmer_contigs.keys()))
#features = np.array([kmer_contigs[name] for name in names], dtype=np.float32)
#np.savez(out_fn + f'feature_k{kmer}_contigs.npz', x=features, name=names)

print('finished')
