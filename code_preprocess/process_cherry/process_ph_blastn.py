import sys
import argparse
import os
import json
import numpy as np
import subprocess
from Bio import SeqIO
from collections import defaultdict
from subprocess import Popen, PIPE
import time
from scipy import sparse, stats
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbimakeblastdbCommandline

def wait_subprocesses(running_procs):
    print('waiting for all subprocesses done...')
    while running_procs:
        time.sleep(10)
        for proc in running_procs:
            retcode = proc.poll()
            if retcode is not None: # Process finished.
                running_procs.remove(proc)
                break
        if retcode != 0:
            print('some child process failed')
    return

def wait_vacant(running_procs, n_processes):
    flag = 0
    while running_procs:
        if len(running_procs) < n_processes:
            break
        time.sleep(5)
        for proc in running_procs:
            retcode = proc.poll()
            if retcode is not None: # Process finished.
                running_procs.remove(proc)
                flag = 1
                break
        if flag == 1:
            break

def process_db(base_dir, n_processes):
    n_processes -= 1
    with open(base_dir + 'host_accs.json') as f:
        host_accs = json.load(f)
    hosts = list(host_accs.keys())
    acc2tid = defaultdict(str)
    for tid, accs in host_accs.items():
        if tid in hosts:
            for acc in accs:
                acc2tid[acc] = tid

    record_dic = defaultdict(list)
    for record in SeqIO.parse(base_dir + 'host.fasta', 'fasta'):
        if len(acc2tid[record.id]) > 0:
            tid = acc2tid[record.id]
            record_dic[tid].append(record)

    base = base_dir + 'cherry/host_database'
    os.makedirs(base, exist_ok=True)
    running_procs = []
    for i, tid in enumerate(hosts):
        records = record_dic[tid]
        for record in records:
            file_base = 'host{}-{}'.format(i, record.id)
            file_name = base + '/' + file_base + '.fasta'
            if not os.path.isfile(file_name):
                SeqIO.write(record, file_name, 'fasta')
            db_fp = base + '/' + file_base + '-db'
            if not os.path.isfile(db_fp + '.ndb'): # one of the database files to create does not exit
                make_blast_cmd = 'makeblastdb -in {} -dbtype nucl -parse_seqids -out {}'.format(file_name, db_fp)
                wait_vacant(running_procs, n_processes)
                print("Creating blast database...")
                running_procs.append(Popen(make_blast_cmd, stdout=PIPE, stderr=PIPE, shell=True))

    wait_subprocesses(running_procs)
    print('finished 1')


def process_blastn(base_dir, n_processes):
    n_processes -= 1
    for idx in range(3):
        # merged phages (created by crispr_split.py)
        name = ['trn', 'val', 'tst'][idx]
        in_fn = base_dir + 'cherry/phage_host/'
        phage_fp = in_fn + 'phages_{}.fasta'.format(name) 
        out_fn = base_dir + 'cherry/phage_host/blastn_result_phage_host_{}/'.format(name)
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)

        with open(base_dir + 'host_accs.json') as f:
            host_accs = json.load(f)
        hosts = list(host_accs.keys())
        acc2tid = defaultdict(str)
        for tid, accs in host_accs.items():
            if tid in hosts:
                for acc in accs:
                    acc2tid[acc] = tid

        base = base_dir + 'cherry/host_database'
        os.path.isdir(base)
        running_procs = []
        for i, tid in enumerate(hosts):
            accs = host_accs[tid]
            for acc in accs:
                file_base = 'host{}-{}'.format(i, acc)
                file_name = base + '/' + file_base + '.fasta'
                db_fp = base + '/' + file_base + '-db'
                assert os.path.isfile(db_fp + '.ndb') # one of the database files to create does exit
                blast_cmd = 'blastn -query {} -db {} -outfmt 6 -out {}.tab -num_threads 4'.format(phage_fp,db_fp,out_fn+file_base)
                wait_vacant(running_procs, n_processes)
                print("Running blastn...")
                running_procs.append(Popen(blast_cmd, stdout=PIPE, stderr=PIPE, shell=True))

        wait_subprocesses(running_procs)
        print(name, 'done')
    print('finished 2')

# merge the network at a higher level (species)
def merge_matrix(ref_mat, row_indices_list):
    # assume ref_mat is in csr form
    n_rows = len(row_indices_list)
    n_cols = ref_mat.shape[1]
    new_mat = sparse.lil_matrix((n_rows, n_cols), dtype=np.int32)
    for i, row_indices in enumerate(row_indices_list):
        if len(row_indices) > 0:
            new_mat[i,:] = ref_mat[row_indices,:].max(0)
    return new_mat

def process_network(base_dir):
    with open(base_dir + 'phage_accs.json') as f:
        phage_accs = json.load(f)
    with open(base_dir + 'host_accs.json') as f:
        host_accs = json.load(f)
    with open(base_dir + 'phage_host_interaction_pos_neg_list.json') as f:
        interactions = json.load(f)
    interactions = [np.array(interactions[i]) for i in range(3)]
    # define query phages and reference hosts
    for idx in range(3):
        phages_all = list(set(np.concatenate([interactions[i] for i in range(0, idx+1)])[:,1]))
        phages_contig_all = [] # for later use
        for phg in phages_all:
            phages_contig_all += phage_accs[phg]
        hosts_all = list(set(np.concatenate([interactions[i] for i in range(0, idx+1)])[:,0]))
        hosts_contig_all = [] # for later use
        for hst in hosts_all:
            hosts_contig_all += host_accs[hst]


        # combine results from blastn-short
        name = ['trn', 'val', 'tst'][idx]
        in_fn = base_dir + 'cherry/phage_host/blastn_result_phage_host_{}/'.format(name)

        # create phage-host network
        mapper = {}
        for i, contig in enumerate(phages_contig_all):
            mapper[contig] = i
        for j, contig in enumerate(hosts_contig_all):
            mapper[contig] = j
        n_rows = len(phages_contig_all)
        n_cols = len(hosts_contig_all)
        mat = sparse.lil_matrix((n_rows, n_cols), dtype=np.int32)
        hosts_all_dic = defaultdict(bool)
        for hst in hosts_all:
            hosts_all_dic[hst] = True
        logT = np.log10(n_rows * n_cols)
        max_sig = 300
        for i, hst in enumerate(host_accs.keys()):
            if hosts_all_dic[hst]:
                for acc in host_accs[hst]:
                    blast_result_fp = in_fn + 'host{}-{}.tab'.format(i, acc)
                    if os.path.isfile(blast_result_fp):
                        with open(blast_result_fp) as f:
                            for line in f.readlines():
                                parse = line.replace("\n", "").split("\t")
                                virus = parse[0]
                                prokaryote = parse[1]
                                evalue = float(parse[10])
                                sig = min(max_sig, np.nan_to_num(-np.log10(evalue) - logT))
                                mat[mapper[virus], mapper[prokaryote]] = sig
                    else:
                        print(blast_result_fp + ' is not found!')

        phages_ind = []
        for phage in phages_all:
            phages_ind.append([mapper[acc] for acc in phage_accs[phage]])
        hosts_ind = []
        for host in hosts_all:
            hosts_ind.append([mapper[acc] for acc in host_accs[host]])
        mat = mat.tocsr()
        new_mat = merge_matrix(mat, phages_ind)
        new_mat = merge_matrix(new_mat.transpose().tocsr(), hosts_ind)
        new_mat = new_mat.transpose().tocsr()

        out_base = base_dir + f'cherry/phage_host/blastn_{name}_'
        sparse.save_npz(out_base + 'contigs_mat.npz', mat)
        sparse.save_npz(out_base + 'species_mat.npz', new_mat)
        with open(out_base+f'contigs_phages.json', 'w') as f:
            json.dump(phages_contig_all, f, indent=2)
        with open(out_base+f'contigs_hosts.json', 'w') as f:
            json.dump(hosts_contig_all, f, indent=2)
        with open(out_base+f'species_phages.json', 'w') as f:
            json.dump(phages_all, f, indent=2)
        with open(out_base+f'species_hosts.json', 'w') as f:
            json.dump(hosts_all, f, indent=2)

        print(name, 'done')
    print('finished 3')

base_dir = '../data_cherry/'
n_processes = 16
process_db(base_dir, n_processes)
process_blastn(base_dir, n_processes)
process_network(base_dir)
print('finished')
