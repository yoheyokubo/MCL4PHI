# code reference: https://github.com/yaozhong/CL4PHI

from collections import defaultdict
from runner import Runner
from data import Data, InteractionDataset, get_dataloaders, get_graphloaders
from evaluator import PrecisionRecallEvaluator, save_results
import pickle
import os
import torch
from torch import nn
import torch.optim as optim
import random
import json
import wandb
import numpy as np
import scipy.sparse as sp
import gc
import itertools

class cnn_module(nn.Module):
    def __init__(self, kernel_size=7, dr=0):
        super(cnn_module, self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=kernel_size, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,128, kernel_size=kernel_size, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dr)
    
        self.fc1 = nn.Linear(4608, 512)

    def forward(self, x): # (bs, 1, 2**6, 2**6)
        x = self.bn1(self.relu(self.conv1(x))) # (bs, 64, 29, 29)
        x = self.bn2(self.relu(self.conv2(x))) # (bs, 128, 12, 12)
        x = self.maxpool(x) # (bs, 128, 6, 6)

        x = self.fc1(torch.flatten(x, 1)) # (bs, 512)
        return x

def euclidean_distance(x0, x1):
    return torch.sqrt(torch.clamp(torch.sum((x0 - x1) ** 2, -1), min=1e-6))

def contrastive_dist(x0, x1, y):
    dist_sq = torch.sum((x0 - x1) ** 2, -1)
    dist = torch.sqrt(torch.clamp(dist_sq, min=1e-6))
    loss = y * dist + (1-y) * torch.clamp(1.0 - dist, min=0.0)
    loss = torch.mean(loss)
    return loss, dist


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        dist_sq = torch.sum((x0 - x1) ** 2, -1)
        dist = torch.sqrt(torch.clamp(dist_sq, min=1e-6))

        loss = y * dist_sq + (1-y) * (torch.clamp(self.margin - dist, min=0.0) ** 2)
        loss = torch.sum(loss) / 2.0 / x0.shape[0]

        return loss, dist

class CL4PHIRunner(Runner):
    def __init__(self, args):
        super().__init__(args)

    def run(self):
        """data loading phase"""
        self.start_timer_dataloading()
        args = self.args
        data = CL4PHIData(args.data_dir, args.granularity, data_type='phd' if 'data' not in vars(args) else args.data, augs=args.aug.split(':'))
        self.dataset_stat = data.dataset_stat
        # interaction dataset to construct loss
        self.dataloaders = get_dataloaders(data, pos_neg_ratios=[-1, -1, -1] + [-1] * (len(self.dataset_stat)-3), batch_size=128, num_workers=4)
        self.end_timer_dataloading()

        """training mode or testing mode"""
        self.start_timer_running()
        model = cnn_module()
        if self.mode == 'TST':
            model.load_state_dict(torch.load(self.model_pretrained))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.criterion = ContrastiveLoss()
        self.evaluator = PrecisionRecallEvaluator(args.granularity)
        if self.mode == 'TRN':
            self.train()
        elif self.mode == 'TST':
            self.test()
        self.end_timer_running()


    def one_epoch(self, epoch, mode, dataloader, query='none'):
        if mode == 0:
            self.model.train()
        else:
            self.model.eval()

        dists = []
        hst_locs_list = []
        phg_locs_list = []
        labels = []

        epoch_loss = 0
        n_samples = 0
        embeds_pu = []
        with torch.autograd.set_grad_enabled(mode==0):
            for hst_locs, phg_locs, label in dataloader['loader']:    
    
                imgs_hst = dataloader['data'].get_item(hst_locs)
                imgs_phg = dataloader['data'].get_item(phg_locs)
                imgs = torch.cat([imgs_hst, imgs_phg])
                bs, k, c, h, w = imgs.shape
                imgs = imgs.reshape(-1, c, h, w).to(self.device)
                outputs = self.model(imgs)
                embeds = outputs.reshape(bs, k, -1)
                embeds_hst, embeds_phg = embeds.chunk(2)
                
                loss, dist = contrastive_dist(embeds_hst[:,0], embeds_phg[:,0], torch.from_numpy(label).to(self.device))
                loss *= bs
                loss += (euclidean_distance(embeds_hst[:,:,None], embeds_hst[:,None]) + \
                         euclidean_distance(embeds_phg[:,:,None], embeds_phg[:,None])).mean()

                bs = len(hst_locs)
                n_samples += bs
                epoch_loss += loss.item() * bs
                
                if mode == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                dists.append(dist.detach().cpu().numpy())
                labels.append(label)
                hst_locs_list.append(hst_locs)
                phg_locs_list.append(phg_locs)

        hst_locs = np.concatenate(hst_locs_list)
        phg_locs = np.concatenate(phg_locs_list)
        lins = np.stack([dataloader['data'].get_lineage(hst_locs), dataloader['data'].get_lineage(phg_locs)], axis=1)
        labels = np.concatenate(labels)
        dists = np.concatenate(dists)

        results = {}
        results['loss'] = epoch_loss / n_samples
        # apply a margin classifier
        results_eval = self.evaluator.evaluate(labels, (dists < 1.0).astype(int), 1 / np.maximum(1e-3, dists), lins, query)
        results.update(results_eval)

        dataloader['data'].reload_local()
        return results


class CL4PHIData(Data):
    def __init__(self, data_dir, granularity, data_type='phd', augs=['0:0']):
        base_path = os.path.join(data_dir, 'cl4phi', '')
        
        """load node features"""
        
        # define the global order in this dataset
        node_feature = np.load(base_path + f'fcgr_k6_d0.0_{granularity}.npz') # corresponding to the original fcgr
        global_ids = node_feature['name']
        mapper = {gid: i for i, gid in enumerate(global_ids)}
        super().__init__(data_dir, granularity, global_ids, data_type=data_type)
        raw_X = node_feature['x'] # ndarray of shape (number of phages and hosts, 2**6, 2**6) for k-mer fcgr image
        # normalization
        trn_mask = [self.trn_ids_dic[loc] for loc in range(len(global_ids))]
        self.set_scaler(raw_X[trn_mask], norm_type='standard') # note that the original cl4phi does not normalize features

        Xs = []
        for aug in augs:
            node_feature = np.load(base_path + f'fcgr_k6_d{aug}_{granularity}.npz')
            mapper_aug = {gid: i for i, gid in enumerate(node_feature['name'])}
            raw_X = node_feature['x'][[mapper_aug[gid] for gid in global_ids]]
            X = torch.from_numpy(self.norm(raw_X))[:,None] # add channel dimension
            Xs.append(X)

        self.X = torch.stack(Xs, dim=1) # (# of phages and hosts, # of augs, 1, 2**6, 2**6)
        print(self.X.shape)
        
    def get_lineage(self, idx):
        return self.lineages[idx]

    def get_item(self, idx):
        return self.X[idx]

    def get_dataset(self, i):
        return CL4PHIInteractionDataset(self.interactions_list[i])


class CL4PHIInteractionDataset(InteractionDataset):
    def __init__(self, interactions):
        super().__init__(interactions)

    def my_collate_fn(self, batch): # e.g., [interactions[0], ...,interactions[bs-1]] where the interctions is ndarray of shape (*,3)
        batched = np.stack(batch)
        hst_locs, phg_locs, label = batched[:,0], batched[:,1], batched[:,2]
        return hst_locs, phg_locs, label
