import os
import json
import numpy as np
from pyfaidx import Fasta
from collections import defaultdict
import math
import random
import sys

random.seed(123)

data_dir = '../data_cherry/'

with open(data_dir + 'phage_host_accs.json') as f:
    phage_host_accs = json.load(f)

# code reference for count_kmers, probablities, and chaos_game_representation: 
# https://github.com/yaozhong/CL4PHI/blob/main/code/fasta2CGR.py
# https://towardsdatascience.com/chaos-game-representation-of-a-genetic-sequence-4681f1a67e14

def count_kmers(sequences, k, dropout=0):
    d = defaultdict(int)
    for sequence in sequences:
        for i in range(len(sequence)-(k-1)):
            if dropout==0 or random.random() > dropout:
                d[sequence[i:i+k]] += 1
        for key in list(d.keys()):
            if "N" in key:
                del d[key]
    return d

def calc_probabilities(kmer_count, k):
    probabilities = defaultdict(float)
    total_counts = max(1, sum(value for value in kmer_count.values()))
    for key, value in kmer_count.items():
        probabilities[key] = float(value) / total_counts
    return probabilities

def chaos_game_representation(probabilities, k):
    array_size = int(math.sqrt(4**k))
    chaos = []
    for i in range(array_size):
        chaos.append([0]*array_size)

    maxx, maxy = array_size, array_size
    posx, posy = 1, 1

    for key, value in probabilities.items():
        for char in reversed(key): # assume key only contains upper characters
            if char == "T":
                posx += maxx / 2
            elif char == "C":
                posy += maxy / 2
            elif char == "G":
                posx += maxx / 2
                posy += maxy / 2
            maxx = maxx / 2
            maxy /= 2

        chaos[int(posy)-1][int(posx)-1] = value
        maxx = array_size
        maxy = array_size
        posx = 1
        posy = 1

    return chaos

def get_FCGRimage(sequences, k, dropout=0):
    kmer_count = count_kmers(sequences, k, dropout=dropout)
    probabilities = calc_probabilities(kmer_count, k)
    return chaos_game_representation(probabilities, k)

#def get_FCGRimage(sequences, k):
#    kmer_count = count_kmers(sequences, k)
#    prob = probabilities(kmer_count, k)
#    return chaos_game_representation(prob, k)

if len(sys.argv) > 0:
    dropout = float(sys.argv[1])
else:
    dropout = 0

wgs = Fasta(data_dir + 'phage_host.fasta')
kmer = 6
images_species = {}
images_contigs = {}
acc_status = defaultdict(bool)
for acc in wgs.keys():
    acc_status[acc] = True
# species level images
for taxid, accs in phage_host_accs.items():
    seqs = [wgs[acc][:].seq.upper() for acc in accs if acc_status[acc]]
    if len(seqs) > 0:
        images_species[taxid] = get_FCGRimage(seqs, kmer, dropout=dropout)
## contig level images
#for acc in wgs.keys():
#    seqs = [wgs[acc][:].seq.upper()]
#    images_contigs[acc] = get_FCGRimage(seqs, kmer)

out_fn = data_dir + 'cl4phi/'
os.makedirs(os.path.dirname(out_fn), exist_ok=True)
names = np.array(list(images_species.keys()))
images = np.array([images_species[name] for name in names], dtype=np.float32)
np.savez(out_fn + f'fcgr_k{kmer}_d{dropout}_species.npz', x=images, name=names)
#names = np.array(list(images_contigs.keys()))
#images = np.array([images_contigs[name] for name in names], dtype=np.float32)
#np.savez(out_fn + f'fcgr_k{kmer}_contigs.npz', x=images, name=names)

print('finished')
