'''
Following codes are adapted from https://github.com/wanchunnie/PB-LKS/blob/main/PBLKS/PBLKS_pred.py
The adaption starts from here.
'''

import numpy as np
from Bio import SeqIO
from itertools import product
import pickle
from tqdm import tqdm
from typing import Dict, List
import os
import random
import sys
from collections import defaultdict

random.seed(123)

# gets all sequence permutations of length 4（repeats included）and makes the index of each kmer
# key order matters here
KMER = 6
KMER_IDX_DICT = {
    ''.join(K): idx for K, idx in zip(product(['A', 'T', 'G', 'C'], repeat=KMER), range(4**KMER))
}

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

def _parse_sequence(sequence, dropout=0):  # shape: [len(sequence) - kmer_with_N, 256]
    '''calculates the number of each type of 4mers(without N)
    the result is orgnized to a 2-dim np array of shape [len(sequence) - kmer_with_N, 256]
    type of 4mer starts with each nt is one-hot encoded on the second dimension
    '''
    index_list = []
    seq_length = len(sequence)
    count = 0
    for i in range(seq_length - KMER + 1):
        kmer = sequence[i: i + KMER]
        if kmer in KMER_IDX_DICT.keys() and (dropout==0 or (random.random() > dropout)): # ignore if this Kmer includes N
            index_list.append(KMER_IDX_DICT[kmer])
            count += 1
    result = np.zeros([count, 4**KMER])
    if count > 0:
        result[np.arange(count), np.array(index_list)] = 1
    return result


def _countkmer(fasta_path, description, dropout=0):
    '''Counts the number of kmer in each sequence in the given fasta file
    Args:
        fasta_path: path to the fasta file
        description: words used infront of progress bar
    Returns:
        kmer_dic: a dictionary, with description of sequence part as key and 
            a list including the number of each kmer as value
    '''
    kmer_dic = {}
    contig_n = 0
    # counts kmer in each sequence longer than 9000
    # scans the sequence in a window of length 9000 and step of length 1800
    # the last 9000nt of the sequence is also scanned separately
    pbar = tqdm(desc=description)
    for record in SeqIO.parse(fasta_path, "fasta"):
        pbar.update()
        seq = record.seq
        seq_length = len(seq)

        if seq_length >= 9000:
            contig_n += 1
            # the kmer type of each base and next 3 nts
            # shape: [sequence length(kmer with N is not included), 256]
            kmer_indexs = _parse_sequence(seq, dropout=dropout)
            start_indexs = np.arange(0, seq_length, 1800)

            for i in start_indexs:
                if i + 9000 <= seq_length:
                    kmer_cnt = np.sum(kmer_indexs[i: i + 9000, :], axis=0)
                    k_num = str(int(i / 1800))
                    kmer_dic[str(contig_n) + '+' + k_num] = kmer_cnt
                else:
                    break

            length = int(seq_length - 9000)
            k_num = str(int(length // 1800 + 1))
            kmer_dic[str(contig_n) + '+' +
                     k_num] = np.sum(kmer_indexs[-9000:, :], axis=0)

    # if none of the sequences in the fasta file has a length beyond 9000nt
    # count mers of length 4 in the longest sequence
    if contig_n == 0:
        max_contig = ''
        contigs = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            #seq = record.seq
            #seq_length = len(seq)
            #if seq_length > len(max_contig):
            #    max_contig = seq
            contigs.append(np.sum(_parse_sequence(record.seq, dropout=dropout), axis=0))
        #print(len(max_contig))
        #kmer_indexs = _parse_sequence(max_contig)
        kmer_dic['1+0'] = sorted(contigs, key=lambda x: np.sum(x), reverse=True)[0]#np.sum(kmer_indexs, axis=0)
    return (kmer_dic)


def _kmerdict2feas(dic_phage: Dict[str, List[int]],
                   dic_bac: Dict[str, List[int]]) -> np.ndarray:
    '''
    find the sequence part pair in dic_phage and dic_bac that has the highest correlation
    in terms of each kemer count 
    subtract the number of each kmer in the chosen pair as model input
    Args:
        dic_phage: a dictionary, with description of sequence part as key and 
            a list including the number of each kmer as value
        dic_bac: similar as dic_phage, a dictionary, with description of sequence part 
            as key and a list including the number of each kmer as value
    Returns:
        the number of each kmer after subtraction
    '''
    p_lst = [p_key for p_key in dic_phage.keys()]
    b_lst = [b_key for b_key in dic_bac.keys()]

    my_corr = {}
    for m in range(len(p_lst)):
        for n in range(len(b_lst)):
            x = dic_phage[p_lst[m]]
            y = dic_bac[b_lst[n]]
            my_corr[p_lst[m] + '_' + b_lst[n]] = np.corrcoef(x, y)[0][-1]

    max_cor = -1
    for test_key in my_corr.keys():
        if my_corr[test_key] > max_cor:
            max_cor = my_corr[test_key]
            max_lne = test_key

    phage_lne = str(max_lne.split('_')[0])
    bac_lne = str(max_lne.split('_')[1])
    # subtract the number of each kmer in the sequence pair as model input
    pb_sub = np.array(dic_phage[phage_lne]) - np.array(dic_bac[bac_lne])
    return pb_sub.reshape(1, -1)


def get_descriptor(bac_path: str,
                   phage_path: str) -> np.ndarray:
    '''
    gets descriptors described in the paper  
    "PB-LKS: a python package for predicting Phage-Bacteria interaction through Local K-mer Strategy"
    Args:
        bac_path: path to the sequence file of Bac(content must be orgnized in fasta format)
        phage_path: path to the sequence file of Phage(content must be orgnized in fasta format)
    Returns:
        descriptors extracted from both sequence files
    Raises:
        FileNotFoundError: if at least one of the sequence file you entered does not exist
        InputFileFormatError: if the content of the sequence file is not orgnized in fasta format
    '''
    if os.path.exists(bac_path) and os.path.exists(phage_path):
        try:
            dic_bac = _countkmer(
                bac_path, description="parsing bacteria sequence")
            dic_phage = _countkmer(
                phage_path, description="parsing phage sequence")
        except Exception:
            raise InputFileFormatError('something wrong happened when parsing sequence file :('
                                       + '\n This error is very likly to be caused by the sequence file content, '
                                       + 'please make sure it is orgnized in fasta format')
        return _kmerdict2feas(dic_phage, dic_bac)
    else:
        raise FileNotFoundError(
            'the sequence file you entered does not exist :(')


'''
The end of the adapted codes
'''
import json
import gzip
from collections import defaultdict
import gc
import pickle
import os

if __name__ == "__main__":

    print(os.getcwd())
    #with open('data_phi/phage_host_interaction_pos_neg.json') as f:
    #    interactions = json.load(f)
    data_dir = '../data_cherry/'
    if len(sys.argv) > 0:
        dropout = float(sys.argv[1])
    else:
        dropout = 0

    with open(data_dir + 'phage_host_accs.json') as f:
        id2accs = json.load(f)
    acc2id = {}
    for tid, accs in id2accs.items():
        for acc in accs:
            acc2id[acc] = tid
    id2accs_rec = defaultdict(dict)
    result_species = {}
    result_contigs = {}

    path = data_dir + 'pblks/species_temp.fasta'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    kterms = [''.join(terms) for terms in product(['A', 'T', 'G', 'C'], repeat=KMER)]
    for record in SeqIO.parse(data_dir + 'phage_host.fasta', "fasta"):
        tid = acc2id[record.id]
        id2accs_rec[tid][record.id] = record
        if len(id2accs_rec[tid]) == len(id2accs[tid]):
            #for acc, rec in id2accs_rec[tid].items():
            #    with open(path, 'w') as f:
            #        SeqIO.write(rec, f, 'fasta')
            #    dic_species = _countkmer(path, description="parsing contig sequence")
            #    result_contigs[acc] = dic_species
            #with open(path, 'w') as f:
            #    SeqIO.write(list(id2accs_rec[tid].values()), f, 'fasta')
            #dic_species = _countkmer(path, description="parsing species sequence", dropout=dropout)
            dic_species = {}
            contig_n = 1
            temp = {}
            for record in id2accs_rec[tid].values():
                if len(record.seq) >= 9000:
                    k_num = 0
                    for i in range(0, len(record.seq), 1800):
                        start_i = i
                        end_i = i + 9000
                        if end_i > len(record.seq):
                            end_i = len(record.seq)
                            start_i = len(record.seq) - 9000
                        kmers = count_kmers([record.seq[start_i:end_i].upper()], KMER, dropout=dropout)
                        temp[str(contig_n)+'+'+str(k_num)] = np.array([kmers[kterm] for kterm in kterms])
                        k_num += 1
                    contig_n += 1
            if contig_n == 1:
                record_max = None
                num_max = 0
                for record in id2accs_rec[tid].values():
                    if len(record.seq) > num_max:
                        record_max = record
                        num_max = len(record.seq)
                k_num = 0
                kmers = count_kmers([record.seq.upper()], KMER, dropout=dropout)
                temp[str(contig_n)+'+'+str(k_num)] = np.array([kmers[kterm] for kterm in kterms])

            result_species[tid] = temp
            id2accs_rec.pop(tid)
            gc.collect()

    #with open('data_cherry/pblks/contigs_dic_pblks_k{}.pkl'.format(KMER),'wb') as f:
    #    pickle.dump(result_contigs, f)
    with open(data_dir + 'pblks/species_dic_pblks_k{}_d{}.pkl'.format(KMER, dropout),'wb') as f: 
        pickle.dump(result_species, f)

    print('finished')
