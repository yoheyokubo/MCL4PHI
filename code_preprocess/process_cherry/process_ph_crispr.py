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

def process_crt(base_dir, n_processes):
    n_processes -= 1
    with open(base_dir + 'host_accs.json') as f:
        host_accs = json.load(f)
    hosts = list(host_accs.keys())
    running_procs = []
    output_fps = []
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

    base = base_dir + 'cherry/host_crisprs'
    crt_path = 'process_cherry/CRT1.2-CLI.jar'
    os.makedirs(base, exist_ok=True)
    for i, tid in enumerate(hosts):
        records = record_dic[tid]
        for record in records:
            file_base = base + '/' + 'host{}-{}'.format(i, record.id)
            file_name = file_base+'.fasta'
            SeqIO.write(record, file_name, 'fasta')
            crt_output_fp = file_base+'.crispr'
            crt_cmd = 'java -cp {} crt {} {}'.format(crt_path, file_name, crt_output_fp)
            os.system(crt_cmd)
            wait_vacant(running_procs, n_processes)
            running_procs.append(Popen(crt_cmd, stdout=PIPE, stderr=PIPE, shell=True))
            output_fps.append((file_base, record.id, crt_output_fp))

    wait_subprocesses(running_procs)
    for file_base, record_id, crt_output_fp in output_fps:
        crispr_rec = []
        with open(crt_output_fp) as file_in:
            for line in file_in:
                if line.startswith('CRISPR'):
                    cnt_crispr = int(line.split(' ')[1]) - 1
                    cnt_spacer = 0
                else:
                    tmp_list = line.split("\t")
                    if len(tmp_list) > 0 and tmp_list[0].isdigit():
                        _ = int(tmp_list[0])
                        if cnt_spacer == 0 and cnt_crispr == 0:
                            print(tmp_list)
                        if len(tmp_list[3]) > 0 and tmp_list[3].isalpha():
                            rec = SeqRecord(Seq(tmp_list[3]), id='{}_CRISPR_{}_SPACER_{}'.format(record_id, cnt_crispr, cnt_spacer), description='')
                            cnt_spacer += 1
                            crispr_rec.append(rec)
        if len(crispr_rec) > 0:
            SeqIO.write(crispr_rec, file_base + "_CRISPRs.fa", 'fasta')

    print('finished 1')

def process_crispr_db(base_dir):

    with open(base_dir + 'host_accs.json') as f:
        host_accs = json.load(f)
    with open(base_dir + 'phage_host_interaction_pos_neg_list.json') as f:
        interactions = json.load(f)
    hosts_all = list(host_accs.keys())
    interactions = [np.array(interactions[i]) for i in range(3)]
    hosts_list = [list(set(interactions[i][:,0])) for i in range(3)]

    # database of host crisprs
    crispr_fn = base_dir + 'cherry/host_crisprs/'

    old_hosts = []
    names = ['trn', 'val', 'tst']
    out_fn = base_dir + "cherry/phage_host/"
    for i in range(3):
        new_hosts = hosts_list[i]
        hosts = list(set(new_hosts + old_hosts))
        crispr_fp = out_fn + 'crisprs_{}.fasta'.format(names[i])
        # load host crisprs from (i=0: trn, i=1: trn+val, i=2: trn+val+tst) dataset
        if not os.path.isfile(crispr_fp):
            records = []
            os.makedirs(os.path.dirname(crispr_fp), exist_ok=True)
            for j, hst in enumerate(hosts_all):
                if hst in hosts:
                    for acc in host_accs[hst]:
                        crispr_inp_fp = crispr_fn+'host{}-{}_CRISPRs.fa'.format(j, acc)
                        if os.path.isfile(crispr_inp_fp):
                            records += list(SeqIO.parse(crispr_inp_fp, 'fasta'))
            SeqIO.write(records, crispr_fp, 'fasta')
            print('crisprs in up to {}  datasets are merged'.format(names[i]))
        crispr_dbs_fp = out_fn + 'crisprs_{}_db'.format(names[i])
        makedb_cmd = NcbimakeblastdbCommandline(dbtype="nucl", input_file=crispr_fp, parse_seqids=True, out=crispr_dbs_fp)
        print("Creating CRISPR database...")
        makedb_cmd()
        old_hosts = hosts

    print('finished 2')

def process_split_blastn(base_dir, n_processes):

    with open(base_dir + 'phage_accs.json') as f:
        phage_accs = json.load(f)
    phages_all = list(phage_accs.keys())
    with open(base_dir + 'phage_host_interaction_pos_neg_list.json') as f:
        interactions = json.load(f)
    interactions = [np.array(interactions[i]) for i in range(3)]

    for idx in range(3):
        # define query phages
        phages_all = list(set(np.concatenate([interactions[i] for i in range(0, idx+1)])[:,1]))
        contigs_all = [] # for later use
        for phg in phages_all:
            contigs_all += phage_accs[phg]

        # merge phage fasta
        name = ['trn', 'val', 'tst'][idx]
        in_fn = base_dir + 'cherry/phage_host/'
        phage_fp = in_fn + 'phages_{}.fasta'.format(name)

        if not os.path.isfile(phage_fp):
            os.makedirs(os.path.dirname(phage_fp), exist_ok=True)
            contigs_dic = defaultdict(bool)
            for contig in contigs_all:
                contigs_dic[contig] = True
            records = []
            for record in SeqIO.parse(base_dir + 'phage.fasta', 'fasta'):
                if contigs_dic[record.id]:
                    records.append(record)
            SeqIO.write(records, phage_fp, 'fasta')
            print(f'phages in up to {name} datasets are merged!')

        # split merged phages for later use of parallel computing
        n_records = 0
        for record in SeqIO.parse(phage_fp, 'fasta'):
            n_records += 1
        n_parallel = n_processes - 1
        batch_size = int(n_records / n_parallel + 1)

        records = []
        count = 0
        file_count = 0
        out_fn = base_dir + 'cherry/phage_host/blastn_result/'
        query_base = out_fn + 'phages_{}'.format(name)
        os.makedirs(os.path.dirname(query_base), exist_ok=True)
        query_fps = []
        for record in SeqIO.parse(phage_fp, 'fasta'):
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
        

        crispr_db_prefix = in_fn + 'crisprs_{}_db'.format(name)
        running_procs = []
        for query_fp in query_fps:
            output_file = os.path.splitext(query_fp)[0] + '_blastn-short.tab'
            print("Running blast-short...")
            crispr_call = NcbiblastnCommandline(query=phage_fp,db=crispr_db_prefix,out=output_file,
                outfmt="6 qseqid sseqid evalue pident length slen", evalue=1,gapopen=10,penalty=-1,
                gapextend=2,word_size=7,dust='no',task='blastn-short',perc_identity=90,num_threads=8)
            running_procs.append(Popen(str(crispr_call), stdout=PIPE, stderr=PIPE, shell=True))

        wait_subprocesses(running_procs)
        print(name, 'done')

    print('finished 3')

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

    for idx in range(3):
        # define query phages and reference hosts
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
        in_fn = base_dir + 'cherry/phage_host/'
        blast_result_fp = in_fn + 'phages_{}_crispr_blastn-short.tab'.format(name)

        _ = subprocess.check_call('cat ' + in_fn + f'blastn_result/phages_{name}_*.tab > ' + blast_result_fp, shell=True)

        # create crispr network
        mapper = {}
        for i, contig in enumerate(phages_contig_all):
            mapper[contig] = i
        for j, contig in enumerate(hosts_contig_all):
            mapper[contig] = j
        n_rows = len(phages_contig_all)
        n_cols = len(hosts_contig_all)
        mat = sparse.lil_matrix((n_rows, n_cols), dtype=np.int32)
        with open(blast_result_fp) as f:
            for line in f.readlines():
                parse = line.replace("\n", "").split("\t")
                virus = parse[0]
                prokaryote = parse[1].rsplit('_', 4)[0]
                ident = float(parse[-3])
                length = float(parse[-2])
                slen = float(parse[-1])
                if length/slen > 0.95 and ident > 0.95:
                    mat[mapper[virus], mapper[prokaryote]] = 1

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

        out_base = base_dir + f'cherry/phage_host/crisprs_{name}_'
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

    print('finished 4')

base_dir = '../data_cherry/'
n_processes = 16
process_crt(base_dir, n_processes)
process_crispr_db(base_dir)
process_split_blastn(base_dir, n_processes)
process_network(base_dir)
print('finished')

