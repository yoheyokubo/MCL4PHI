# code reference: https://github.com/wanchunnie/PB-LKS

from runner import Runner
from data import Data, InteractionDataset, get_dataloaders
from evaluator import PrecisionRecallEvaluator, save_results
from sklearn.ensemble import RandomForestClassifier
import wandb
import json
import pickle
import torch
import os
import numpy as np
import itertools

class PBLKSRunner(Runner):
    def __init__(self, args):
        super().__init__(args)

    def run(self):
        """data loading phase"""
        self.start_timer_dataloading()
        args = self.args
        data = PBLKSData(args.data_dir, args.granularity, data_type='phd' if 'data' not in vars(args) else args.data)
        self.data = data
        self.dataset_stat = data.dataset_stat
        # interaction dataset to construct loss
        self.dataloaders = get_dataloaders(data, pos_neg_ratios=[1.0, 1.0, -1] + [-1] * (len(self.dataset_stat)-3), batch_size=128, num_workers=4)
        self.end_timer_dataloading()

        """training mode or testing mode"""
        self.start_timer_running()
        self.evaluator = PrecisionRecallEvaluator(args.granularity)
        if self.mode == 'TRN':
            self.train_pblks(args.seed, os.path.splitext(self.model_save_path)[0] + '.pkl')
        elif self.mode == 'TST':
            self.test_pblks()
        self.end_timer_running()

    def test_pblks(self):
        with open(self.model_pretrained, 'rb') as f:
            cls = pickle.load(f)
        results = {}
        for i in range(2, len(self.dataset_stat)):
            _, _, _, results_tst = self.one_epoch_pblks(i, self.dataloaders[i], model=cls, query=self.dataset_stat[i]['query'])
            results.update({self.dataset_stat[i]['name']+'_'+k:v for k,v in results_tst.items()})
        dic = save_results(results, self)
        wandb.log(dic)

    def train_pblks(self, seed, path):
        X_trn, Y_trn, lins_trn, _ = self.one_epoch_pblks(0, self.dataloaders[0])
        X_val, Y_val, lins_val, _ = self.one_epoch_pblks(1, self.dataloaders[1])
        X = np.concatenate([X_trn, X_val]) # training and validation datasets (each contain an equall number of negatives) are used for training RF
        Y = np.concatenate([Y_trn, Y_val])
        cls = RandomForestClassifier(random_state=seed)
        cls.fit(X, Y)
        results = {}
        _, _, _, results_trn = self.one_epoch_pblks(0, self.dataloaders[0], model=cls, X=X_trn, Y=Y_trn, lins=lins_trn, query=self.dataset_stat[0]['query'])
        _, _, _, results_val = self.one_epoch_pblks(1, self.dataloaders[1], model=cls, X=X_val, Y=Y_val, lins=lins_val, query=self.dataset_stat[1]['query'])
        results.update({'trn_'+k:v for k,v in results_trn.items()})
        results.update({'val_'+k:v for k,v in results_val.items()})
        wandb.log({k:v for k,v in results.items() if isinstance(v, float)})
        with open(path, 'wb') as f:
            pickle.dump(cls, f)
        
        # temporarily
        #results = {}
        #_, _, _, results_tst = self.one_epoch_pblks(2, self.dataloaders[2], model=cls, query=self.dataset_stat[2]['query'])
        #results.update({self.dataset_stat[2]['name']+'_'+k:v for k,v in results_tst.items()})
        #dic = save_results(results, self)
        #wandb.log(dic)
        return

    def get_descriptor(self, x, y):
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
        descriptor = (x[ind[0]] - y[ind[1]]).reshape(1,-1) # pahge - host
        return descriptor, np.max(corr, axis=None)

    def one_epoch_pblks(self, mode, dataloader, model=None, query='none', X=None, Y=None, lins=None):
        loader = dataloader['loader']
        data = dataloader['data']
        if X is None or Y is None or lins is None:
            descriptors = []
            hst_locs_list = []
            phg_locs_list = []
            labels = []
            count = 0
            for hst_locs, phg_locs, label in loader:
                descs = data.get_descriptors(hst_locs, phg_locs)
                if descs is None:
                    kmers_hst = data.get_item(hst_locs)
                    kmers_phg = data.get_item(phg_locs)
                    for kmer_hst, kmer_phg in zip(kmers_hst, kmers_phg):
                        descriptor, _ = self.get_descriptor(kmer_phg, kmer_hst)
                        descriptors.append(descriptor)
                else:
                    descriptors.append(descs)
                hst_locs_list.append(hst_locs)
                phg_locs_list.append(phg_locs)
                labels.append(label)
            hst_locs = np.concatenate(hst_locs_list)
            phg_locs = np.concatenate(phg_locs_list)
            lins = np.stack([data.get_lineage(hst_locs), data.get_lineage(phg_locs)], axis=1)
            labels = np.concatenate(labels)
            X = np.concatenate(descriptors)
            Y = labels

        results = {}
        if model is not None:
            probs = model.predict_proba(X)[:,1]
            results = self.evaluator.evaluate(Y, (probs > 0.5).astype(int), probs, lins, query)
        print('mode {} finished'.format(mode))
        return X, Y, lins, results


class PBLKSData(Data):
    def __init__(self, data_dir, granularity, data_type='phd'):
        with open(os.path.join(data_dir, 'pblks', f'{granularity}_dic_pblks_k6.pkl'), 'rb') as f:
            dic_kmer = pickle.load(f)
        # define the global order of ids (taxids or accession numbers) over this dataset
        global_ids = list(dic_kmer.keys())
        self.X_kmer = [np.array(list(dic_kmer[gid].values()), dtype=np.float32) for gid in global_ids]
        if os.path.isfile(os.path.join(data_dir, 'pblks', f'{granularity}_network.npz')):
            data = np.load(os.path.join(data_dir, 'pblks', f'{granularity}_network.npz'))
            mapper = {gid: i for i, gid in enumerate(global_ids)}
            self.descriptors_loc = {(hst_loc, phg_loc): i for i, (phg_loc, hst_loc) in enumerate(np.vectorize(mapper.get)(data['edge_index']).T)}
            #hst_ids = set(global_ids[hst_loc] for hst_loc, _ in self.descriptors_loc.keys())
            #phg_ids = set(global_ids[hst_loc] for hst_loc, _ in self.descriptors_loc.keys())
            #print(set(global_ids) - (hst_ids | phg_ids), 'are not in the networks')
            #print(len(hst_ids) * len(phg_ids) - len(data['descriptor']), 'combinations are missing in the network itself')
            self.descriptors = data['descriptor']
        else:
            self.descriptors = None
        super().__init__(data_dir, granularity, global_ids, data_type=data_type)

    def get_lineage(self, idx):
        return self.lineages[idx]

    def get_item(self, idx):
        return [self.X_kmer[i] for i in idx]

    def get_descriptors(self, hst_locs, phg_locs):
        if self.descriptors is None:
            return None
        else:
            idx = [self.descriptors_loc[(hst_loc, phg_loc)] for hst_loc, phg_loc in zip(hst_locs, phg_locs)]
            return self.descriptors[idx]

    def get_dataset(self, i):
        return PBLKSInteractionDataset(self.interactions_list[i])

class PBLKSInteractionDataset(InteractionDataset):
    def __init__(self, interactions):
        super().__init__(interactions)

    def my_collate_fn(self, batch):
        batched = np.stack(batch)
        hst_locs, phg_locs, label = batched[:,0], batched[:,1], batched[:,2]
        return hst_locs, phg_locs, label
