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
from Bio.Blast.Applications import NcbimakeblastdbCommandline, NcbiblastnCommandline

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

def process_db(base_dir):
    
    # make blastn database
    out_fn = base_dir + 'cherry/phage_phage_identity'
    os.makedirs(out_fn, exist_ok=True)
    names = ['trn', 'val', 'tst']
    for name in names:
        phage_fp = base_dir + 'cherry/phage_host/phages_{}.fasta'.format(name) # from phage-host-crispr
        db_fp = out_fn + '/' + 'phages_{}_db'.format(name)
        if not os.path.isfile(db_fp + '.ndb'):
            make_blast_cmd = 'makeblastdb -in {} -dbtype nucl -parse_seqids -out {}'.format(phage_fp, db_fp)
            print("Creating blast database...")
            _ = subprocess.check_call(make_blast_cmd, shell=True)

    # data split
    out_base = out_fn + '/blastn_result'
    os.makedirs(out_base, exist_ok=True)
    with open(base_dir + 'phage_accs.json') as f:
        phage_accs = json.load(f)
    phages = list(phage_accs.keys())
    mapper = {tid: i for i, tid in enumerate(phages)}
    acc2tid_dic = defaultdict(str)
    for phg in phages:
        for acc in phage_accs[phg]:
            acc2tid_dic[acc] = phg

    for record in SeqIO.parse(base_dir + 'phage.fasta', 'fasta'):
        if len(acc2tid_dic[record.id]) > 0:
            tid = acc2tid_dic[record.id]
            SeqIO.write(record, out_base + '/phage_{}_{}.fasta'.format(mapper[tid], record.id), 'fasta')
    
    print('finished 1')


def process_blastn(base_dir, n_processes, minimal=True):
    n_processes -= 1
    for idx in range(3):
        # merged phages (created by crispr_split.py)
        name = ['trn', 'val', 'tst'][idx]
        in_fn = base_dir + 'cherry/phage_phage_identity/'
        if not minimal:
            db_fp = in_fn + 'phages_{}_db'.format(name)
        else:
            db_fp = in_fn + 'phages_{}_db'.format(['trn', 'trn', 'val'][idx])
        out_fn = base_dir + 'cherry/phage_phage_identity/blastn_result/'

        with open(base_dir + 'phage_accs.json') as f:
            phage_accs = json.load(f)
        phages = list(phage_accs.keys())
        with open(base_dir + 'phage_host_interaction_pos_neg_list.json') as f:
            interactions = json.load(f)
        interactions = [np.array(interactions[i]) for i in range(3)]
        phage_filter = defaultdict(bool)

        if minimal and idx > 0: # note: phage-by-phage blastn will take much time, so in minimal mode just calculate for phages only used in CHERRY's prediction
            interactions_merged = np.concatenate(interactions[0:idx])
            phages_seen = np.unique(interactions_merged[interactions_merged[:,2] == '1'][:,1])
            phages_query = np.unique(interactions[idx][:,1])
            mask = np.isin(phages_query, phages_seen, invert=True)
            phages_all = phages_query[mask] # phages unseen in previous positive interactions
            for phage in phages_all:
                phage_filter[phage] = True
        else:
            phages_all = np.unique(np.concatenate(interactions[0:idx+1])[:,1])
            for phage in phages_all:
                phage_filter[phage] = True

        acc2tid = defaultdict(str)
        for tid, accs in phage_accs.items():
            if tid in phages:
                for acc in accs:
                    acc2tid[acc] = tid

        running_procs = []
        for i, tid in enumerate(phages):
            if not phage_filter[tid]:
                continue
            accs = phage_accs[tid]
            for acc in accs:
                file_base = out_fn + 'phage_{}_{}'.format(i, acc)
                phage_fp = file_base + '.fasta'
                out_fp = file_base + '_{}.tab'.format(name)
                virus_call = NcbiblastnCommandline(query=phage_fp,db=db_fp,out=out_fp,
                        outfmt="6 qseqid sseqid evalue pident length qlen", evalue=1e-10,gapopen=10,penalty=-1,
                        gapextend=2,word_size=7,dust='no', task='megablast',perc_identity=90,num_threads=16)
                #blast_cmd = 'blastn -query {} -db {} -outfmt 6 -out {} -num_threads 4'.format(phage_fp,db_fp,out_fp)
                wait_vacant(running_procs, n_processes)
                running_procs.append(Popen(str(virus_call), stdout=PIPE, stderr=PIPE, shell=True))
                print("Running blastn...")
                #_ = subprocess.check_call(blast_cmd, shell=True)
                #virus_call()

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

    
def process_network(base_dir, n_processes):
    with open(base_dir + 'phage_accs.json') as f:
        phage_accs = json.load(f)
    with open(base_dir + 'phage_host_interaction_pos_neg_list.json') as f:
        interactions = json.load(f)
    interactions = [np.array(interactions[i]) for i in range(3)]

    for idx in range(3):
        # define all phages
        phages_all = list(set(np.concatenate([interactions[i] for i in range(0, idx+1)])[:,1]))
        phages_contig_all = [] # for later use
        for phg in phages_all:
            phages_contig_all += phage_accs[phg]

        phages_seen = np.unique(interactions[0][interactions[0][:,2] == '1'][:,1])
        if idx == 2:
            phages_seen = np.unique(np.concatenate([phages_seen, np.unique([interactions[1][interactions[1][:,2] == '1'][:,1]])]))
        phages_query = np.unique(interactions[idx][:,1])
        mask = np.isin(phages_query, phages_seen, invert=True)
        phages_unseen = phages_query[mask]
        print(len(phages_unseen))

        # folder for the results from blastn
        name = ['trn', 'val', 'tst'][idx]
        in_fn = base_dir + 'cherry/phage_phage_identity/blastn_result/'
        
        # create phage-phage network
        mapper = {}
        for i, contig in enumerate(phages_contig_all):
            mapper[contig] = i
        n_rows = n_cols = len(phages_contig_all)
        mat = sparse.lil_matrix((n_rows, n_cols), dtype=np.int32)
        phages_unseen_dic = defaultdict(bool)
        for phg in phages_unseen:
            phages_unseen_dic[phg] = True
        for i, phg in enumerate(phage_accs.keys()):
            if phages_unseen_dic[phg]:
                for acc in phage_accs[phg]:
                    blast_result_fp = in_fn + 'phage_{}_{}_{}.tab'.format(i, acc, name)
                    if os.path.isfile(blast_result_fp):
                        with open(blast_result_fp) as f:
                             for line in f.readlines():
                                 parse = line.replace("\n", "").split("\t")
                                 virus = parse[0]
                                 virus_ref = parse[1].split('|', 1)[1].rsplit('|')[0]
                                 ident = float(parse[-3])
                                 length = float(parse[-2])
                                 qlen = float(parse[-1])
                                 if length/qlen > 0.9 and ident > 0.9:
                                     mat[mapper[virus], mapper[virus_ref]] = 1
                                     mat[mapper[virus_ref], mapper[virus]] = 1
                    else:
                        print(blast_result_fp + ' is not found!')

        phages_ind = []
        for phage in phages_all:
            phages_ind.append([mapper[acc] for acc in phage_accs[phage]])
        mat = mat.tocsr()
        new_mat = merge_matrix(mat, phages_ind)
        new_mat = merge_matrix(new_mat.transpose().tocsr(), phages_ind)
        new_mat = new_mat.transpose().tocsr()

        out_base = base_dir + f'cherry/phage_phage_identity/blastn_{name}_'
        sparse.save_npz(out_base + 'contigs_mat.npz', mat)
        sparse.save_npz(out_base + 'species_mat.npz', new_mat)
        with open(out_base+f'contigs_phages.json', 'w') as f:
            json.dump(phages_contig_all, f, indent=2)
        with open(out_base+f'species_phages.json', 'w') as f:
            json.dump(phages_all, f, indent=2)

        print(name, 'done')
        
    print('finished 3')

base_dir = '../data_cherry/'
n_processes = 16
process_db(base_dir)
process_blastn(base_dir, n_processes)
process_network(base_dir, n_processes)
print('finished')
