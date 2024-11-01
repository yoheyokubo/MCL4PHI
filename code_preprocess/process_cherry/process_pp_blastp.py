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
from utils import make_protein_clusters_mcl

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

def process_prodigal(base_dir, n_processes=16):
    n_processes -= 1
    with open(base_dir + 'phage_accs.json') as f:
        phage_accs = json.load(f)
    phages = list(phage_accs.keys())
    batch_size = int(len(phages) / n_processes) + 1

    running_procs = []
    for idx in range(n_processes):
        start_loc = idx * batch_size
        end_loc = min(len(phages), start_loc + batch_size)
        phages_target = phages[start_loc:end_loc]
        acc2tid = defaultdict(str)
        for tid, accs in phage_accs.items():
            if tid in phages_target:
                for acc in accs:
                    acc2tid[acc] = tid
        record_dic = defaultdict(list)
        for record in SeqIO.parse(base_dir + 'phage.fasta', 'fasta'):
            if len(acc2tid[record.id]) > 0:
                tid = acc2tid[record.id]
                record_dic[tid].append(record)        
        base = base_dir + 'cherry/phage_annotations'
        os.makedirs(base, exist_ok=True)
        
        for i, tid in enumerate(phages_target):
            records = record_dic[tid]
            for record in records:
                file_base = base + '/' + 'phage{}-{}'.format(start_loc + i, record.id)
                file_name = file_base+'.fasta'
                SeqIO.write(record, file_name, 'fasta')
                prodigal_cmd = 'prodigal -i ' + file_name + ' -a ' + file_base+'-proteins.faa'+ ' -o ' + file_base + '.output' + ' -f gff -p meta'
                print("Running prodigal...")
                running_procs.append(Popen(prodigal_cmd, stdout=PIPE, stderr=PIPE, shell=True))

    wait_subprocesses(running_procs)
    print('finished 1')

def process_diamond_db(base_dir):
    with open(base_dir + 'phage_accs.json') as f:
        phage_accs = json.load(f)
    with open(base_dir + 'phage_host_interaction_pos_neg_list.json') as f:
        interactions = json.load(f)
    phages_all = list(phage_accs.keys()) # note that some phages in phages_all are not in interactions in PHD dataset
    interactions = [np.array(interactions[i]) for i in range(3)]
    phages_list = [list(set(interactions[i][:,1])) for i in range(3)]

    # database of annotated proteins
    protein_fn = base_dir + 'cherry/phage_annotations/'

    old_phages = []
    names = ['trn', 'val', 'tst']
    out_fn = base_dir + "cherry/phage_phage/"
    for i in range(3):
        new_phages = phages_list[i]
        phages = list(set(new_phages + old_phages))
        protein_fp = out_fn + 'proteins_{}.fasta'.format(names[i])
        # load phage proteins from (i=0: trn, i=1: trn+val, i=2: trn+val+tst) dataset
        if not os.path.isfile(protein_fp):
            records = []
            os.makedirs(os.path.dirname(protein_fp), exist_ok=True)
            for j, phg in enumerate(phages_all):
                if phg in phages:
                    for acc in phage_accs[phg]:
                        records += list(SeqIO.parse(protein_fn+'phage{}-{}-proteins.faa'.format(j, acc), 'fasta'))
            SeqIO.write(records, protein_fp, 'fasta')
            print('proteins in up to {}  datasets are merged'.format(names[i]))
        protein_inp_fp = protein_fp
        protein_dbs_fp = out_fn + 'database_{}.dmnd'.format(names[i])
        make_diamond_cmd = 'diamond makedb --threads 8 --in {} -d {}'.format(protein_inp_fp, protein_dbs_fp)
        print("Creating Diamond database...")
        _ = subprocess.check_call(make_diamond_cmd, shell=True)
        old_phages = phages
    print('finished 2')

def process_diamond_split_blastp(base_dir, n_processes, diamond_option='--sensitive'):
    # note that diamond_option is recommended to be set as '--fast' for such a large dataset as PHD-dataset

    with open(base_dir + 'phage_accs.json') as f:
        phage_accs = json.load(f)
    phages_all = list(phage_accs.keys())
    with open(base_dir + 'phage_host_interaction_pos_neg_list.json') as f:
        interactions = json.load(f)
    interactions = [np.array(interactions[i]) for i in range(3)]

    running_procs = []
    for idx in range(3):
        # run diamond blastp
        name = ['trn', 'val', 'tst'][idx]
        in_fn = base_dir + 'cherry/phage_phage/'
        protein_inp_fp = in_fn + 'proteins_{}.fasta'.format(name)
        protein_dbs_fp = in_fn + 'database_{}.dmnd'.format(name)

        n_records = 0
        for record in SeqIO.parse(protein_inp_fp, 'fasta'):
            n_records += 1
        n_parallel = n_processes - 1
        batch_size = int(n_records / n_parallel + 1)

        records = []
        count = 0
        file_count = 0
        out_fn = base_dir + 'cherry/phage_phage/blastp_result/'
        query_base = out_fn + 'proteins_{}'.format(name)
        os.makedirs(os.path.dirname(query_base), exist_ok=True)
        
        query_fps = []
        for record in SeqIO.parse(protein_inp_fp, 'fasta'):
            records.append(record)
            count += 1
            if count >= batch_size:
                query_fp = query_base + '_phage_{}.fasta'.format(file_count)
                SeqIO.write(records, query_fp, 'fasta')
                records = []
                count = 0
                file_count += 1
                query_fps.append(query_fp)

        if len(records) > 0:
            query_fp = query_base + '_phage_{}.fasta'.format(file_count)
            SeqIO.write(records, query_fp, 'fasta')
            query_fps.append(query_fp)

        fps = []
        fps_brief = []
        for query_fp in query_fps:
            blast_result_fp = os.path.splitext(query_fp)[0] + '_self-diamond.tab'
            blast_result_brief_fp = os.path.splitext(query_fp)[0] + '_self-diamond.tab.abc'
            print("Running Diamond...")
            diamond_cmd = 'diamond blastp --threads 8 {} -d {} -q {} -o {}'.format(diamond_option, protein_dbs_fp, query_fp, blast_result_fp)
            running_procs.append(Popen(diamond_cmd, stdout=PIPE, stderr=PIPE, shell=True))
            
            fps.append(blast_result_fp)
            fps_brief.append(blast_result_brief_fp)

        wait_subprocesses(running_procs)
        for fp, fp_brief in zip(fps, fps_brief):
             _ = subprocess.check_call("awk '$1!=$2 {{print $1,$2,$11}}' {0} > {1}".format(fp, fp_brief), shell=True)
        print(name, 'done')
        
    print('finished 3')

def process_mcl(base_dir):
    with open(base_dir + 'phage_accs.json') as f:
        phage_accs = json.load(f)
    with open(base_dir + 'phage_host_interaction_pos_neg_list.json') as f:
        interactions = json.load(f)
    interactions = [np.array(interactions[i]) for i in range(3)]

    for idx in range(3):
        # define the order of phages (and contigs)
        phages_all = list(set(np.concatenate([interactions[i] for i in range(0, idx+1)])[:,1]))
        contigs_all = [] # for later use
        for phg in phages_all:
            contigs_all += phage_accs[phg]

        # combine results from diamond blastp
        name = ['trn', 'val', 'tst'][idx]
        out_fn = base_dir + f"cherry/phage_phage/proteins_{name}_"
        blast_result_brief_fp = out_fn + f'self-diamond.tab.abc'

        _ = subprocess.check_call(f"cat data_phi/cherry/phage_phage/blastp_result/proteins_{name}_*.abc > " + blast_result_brief_fp, shell=True)

        # do markov clustering
        pc_overlap, pc_penalty, pc_haircut, pc_inflation = 0.8, 2.0, 0.1, 2.0
        pcs_fp = make_protein_clusters_mcl(blast_result_brief_fp, out_fn, pc_inflation)
        print(name, 'done clustering')


        pcs_fp = out_fn + 'merged_mcl20.clusters'
        # load and build protein clusters at the lowest level (protein ids by protein clusters sparse matrix)
        with open(pcs_fp) as f:
            c = [line.rstrip("\n").split("\t") for line in f]
        c = [x for x in c if len(x) > 0]
        data = np.ones(sum(len(x) for x in c), dtype=np.int32)
        row_ids, col_ids = [], []
        row_names = []
        count = 0
        for col_id, x in enumerate(c):
            for xi in x:
                row_names.append(xi)
                row_ids.append(count)
                col_ids.append(col_id)
                count += 1
        if count != len(row_names):
            print('some proteins probably belong to multiple clusters!')
            exit(1)
        row = np.array(row_ids)
        col = np.array(col_ids)
        data = np.ones(count, dtype=np.int32)
        protein2cluster = sparse.coo_matrix((data, (row, col)), shape=(len(row_names), len(c))).tocsr()
        row_names_dic = {row_name:i for i, row_name in enumerate(row_names)}

        # merge the clusters at higher levels (contigs and species)
        def merge_matrix(ref_mat, row_indices_list):
            # assume ref_mat is in csr form
            n_rows = len(row_indices_list)
            n_cols = ref_mat.shape[1]
            new_mat = sparse.lil_matrix((n_rows, n_cols), dtype=np.int32)
            for i, row_indices in enumerate(row_indices_list):
                if len(row_indices) > 0:
                    new_mat[i,:] = ref_mat[row_indices,:].max(0)
            return new_mat

        contigs_dic = defaultdict(list)
        for row_name in row_names:
            contig = row_name.rsplit("_", 1)[0]
            contigs_dic[contig].append(row_names_dic[row_name])
        contigs_ind = [contigs_dic[contig] for contig in contigs_all]
        contig2cluster = merge_matrix(protein2cluster, contigs_ind)
        contigs_pos_dic = {contig: i for i, contig in enumerate(contigs_all)}
        phages_ind = []
        for phage in phages_all:
            phages_ind.append([contigs_pos_dic[acc] for acc in phage_accs[phage]])
        species2cluster = merge_matrix(contig2cluster.tocsr(), phages_ind)

        # construct hypergeometric similarity network
        def hypergeometric_network(file_path, ref_mat, thres=1, max_sig=300):
            # assume ref_mat is in csr form
            n_units, n_pcs_total = ref_mat.shape
            n_pcs = ref_mat.sum(1)
            T = 0.5 * n_units * (n_units - 1)
            logT = np.log10(T)
            new_mat = ref_mat @ ref_mat.transpose()
            S = sparse.lil_matrix((n_units, n_units), dtype=np.float32)
            total_c = float(new_mat.getnnz())
            i = 0 # counter for display
            for A, B in zip(*new_mat.nonzero()): # For A & B sharing at least one protein cluster
                if A != B:
                    a, b = n_pcs[A,0], n_pcs[B,0]
                    pval = stats.hypergeom.sf(new_mat[A, B] - 1, n_pcs_total, a, b)
                    sig = min(max_sig, np.nan_to_num(-np.log10(pval) - logT))
                    if sig > thres:
                        S[min(A, B), max(A, B)] = sig
                    i += 1
                    if i % 1000 == 0:
                        sys.stdout.write(".")
                    if i % 10000 == 0:
                        sys.stdout.write("{:6.2%} {}/{}\n".format(i / total_c, i, total_c))
            S += S.T
            S = S.tocsr()
            if len(S.data) != 0:
                print("\nHypergeometric contig-similarity network:\n {0:10} units,\n {1:10} edges (min:{2:.2} "
                    "max: {3:.2}, threshold was {4})".format(n_units, S.getnnz(), S.data.min(), S.data.max(), thres))
            else:
                raise ValueError("No edge in the similarity network !")
            sparse.save_npz(file_path, S)

        hypergeometric_network(out_fn+f'contigs_mat.npz', contig2cluster.tocsr())
        hypergeometric_network(out_fn+f'species_mat.npz', species2cluster.tocsr())
        with open(out_fn+f'contigs.json', 'w') as f:
            json.dump(contigs_all, f, indent=2)
        with open(out_fn+f'species.json', 'w') as f:
            json.dump(phages_all, f, indent=2)

        print(name, 'done creating network')

    print('finished 4')

base_dir = '../data_cherry/'
# assume 'phage_accs.json', 'phage.fasta', and 'phage_host_interaction_pos_neg_list.json' are in the base_dir
n_processes = 16
process_prodigal(base_dir, n_processes)
process_diamond_db(base_dir)
process_diamond_split_blastp(base_dir, n_processes)
process_mcl(base_dir)

print('finished')

