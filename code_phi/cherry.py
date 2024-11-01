# code reference: https://github.com/KennthShang/CHERRY

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
import gc
import scipy.sparse as sp
from torch_geometric.data import Data as HomoData
from torch_geometric import transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix

class cherry_module(nn.Module):
    def __init__(self, in_channels=4**4, out_channels=4**4, hidden_dim_1=128, hidden_dim_2=32):
        super().__init__()
        self.encoder = GCNConv(in_channels, out_channels)
        self.decoder = nn.Sequential(
                        nn.Linear(out_channels, hidden_dim_1),
                        nn.ReLU(),
                        nn.Linear(hidden_dim_1, hidden_dim_2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim_2, 1)
                    )

    def forward(self, x, edge_index, edge_label_index):
        x = self.encoder(x, edge_index)
        prob = self.decoder(x[edge_label_index[1]] - x[edge_label_index[0]]) # phage - host
        return prob.reshape(-1)

class CHERRYRunner(Runner):
    def __init__(self, args):
        super().__init__(args)

    def run(self):
        """data loading phase"""
        self.start_timer_dataloading()
        args = self.args
        self.data = CHERRYData(args.data_dir, args.granularity, args.cherry_multihost, data_type='phd' if 'data' not in vars(args) else args.data)
        self.dataset_stat = self.data.dataset_stat
        # interaction dataset to construct loss
        self.dataloaders = get_graphloaders(self.data, pos_neg_ratios=[1.0, -1, -1] + [-1] * (len(self.dataset_stat)-3), batch_size=128, num_workers=4, graph_type='homo')
        self.end_timer_dataloading()

        """training mode or testing mode"""
        self.start_timer_running()
        model = cherry_module()
        if self.mode == 'TST':
            model.load_state_dict(torch.load(self.model_pretrained))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.evaluator = PrecisionRecallEvaluator(args.granularity)
        if self.mode == 'TRN':
            self.train(lr=1e-2)
        elif self.mode == 'TST':
            self.test()
        self.end_timer_running()


    def reload_trn_dataloader(self):
        self.dataloaders[0] = get_graphloaders(self.data, pos_neg_ratios=[1.0], batch_size=128, num_workers=16, graph_type='homo')[0]
        return


    def one_epoch(self, epoch, mode, dataloader, query='none'):
        if mode == 0:
            self.model.train()
        else:
            self.model.eval()

        probs = []
        input_ids = []

        epoch_loss = 0
        n_samples = 0
        with torch.autograd.set_grad_enabled(mode==0):
            for data in dataloader['loader']:    
    
                edge_label_index = data.edge_label_index
                logit = self.model(data.x.to(self.device), data.edge_index.to(self.device), edge_label_index)

                loss = self.criterion(logit, data.edge_label.float().to(self.device))
                bs = len(edge_label_index)
                n_samples += bs
                epoch_loss += loss.item() * bs
                
                if mode == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                probs.append(torch.sigmoid(logit).detach().cpu().numpy())
                input_ids.append(data.input_id.numpy()) # global id for edge_label_index

        input_ids = np.concatenate(input_ids)
        edge_label_index = dataloader['dataset'].graph_data.edge_label_index[:,input_ids].numpy()
        labels = dataloader['dataset'].graph_data.edge_label[input_ids].numpy()
        pos_hst, pos_phg = edge_label_index
        lins = np.stack([dataloader['data'].get_lineage(pos_hst), dataloader['data'].get_lineage(pos_phg)], axis=1)
        probs = np.concatenate(probs)

        # add deterministic predictions
        phg2hst_dic = dataloader['dataset'].phg2hst_dic
        for i in range(len(probs)):
            #print(pos_hst[i])
            #print(phg2hst_dic[pos_phg[i]])
            if pos_hst[i] in phg2hst_dic[pos_phg[i]]:
                probs[i] = 1

        results = {}
        results['loss'] = epoch_loss / n_samples
        # we modified cherry's original decision threshold (topk for test_phage-train_host and 0.98 for test_host-train_host)
        results_eval = self.evaluator.evaluate(labels, (probs > 0.5).astype(int), probs, lins, query)
        results.update(results_eval)
        
        # reload training dataloader
        if mode == 0 and self.args.cherry_period > 0 and epoch % self.args.cherry_period == 0:
            self.reload_trn_dataloader()
            gc.collect()

        return results


class CHERRYData(Data):
    def __init__(self, data_dir, granularity, multihost, data_type='phd'):
        base_path = os.path.join(data_dir, 'cherry', '')
        self.multihost = multihost
        
        """load node features"""
        node_feature = np.load(base_path + f'feature_k4_{granularity}.npz')
        # define the global order in this dataset
        global_ids = node_feature['name']
        mapper = {gid: i for i, gid in enumerate(global_ids)}
        super().__init__(data_dir, granularity, global_ids, data_type=data_type)
        raw_X = node_feature['x'] # ndarray of shape (number of phages and hosts, 265) for k-mer counts
        # convert counts to frequencies (unlike offifical cherry implementation) to offset the effect of sequence lengths on feature magnitude
        raw_X = raw_X / np.maximum(1, np.sum(raw_X, axis=1, keepdims=True))
        # normalization
        trn_mask = [self.trn_ids_dic[loc] for loc in range(len(global_ids))]
        self.set_scaler(raw_X[trn_mask], norm_type='minmax') # note that cherry normalize features for training and testing phages and hosts, respectively (four combinations)
        X = torch.from_numpy(self.norm(raw_X))
        self.node_feature = X
        
        """load edge features"""
        edge_types = ['phage_phage_blastn', 'phage_phage_blastp', 'phage_host_blastn', 'phage_host_crisprs']
        self.edge_features = []
        for i, mode in enumerate(['trn', 'val', 'tst']):
            temp = {}
            for edge_type in edge_types:
                graph = np.load(base_path + f'{edge_type}_{mode}_{granularity}.npz')
                edge_index = np.vectorize(mapper.get)(graph['edge_index']) if graph['edge_index'].shape[1] > 0 else graph['edge_index'].astype(np.int32) # map accession number (str) to location (int) in node features
                temp[edge_type] = edge_index
                #edge_attr = graph['edge_attr'] # cherry doesn't take edge attributes into account
            temp['edge_index'] = np.unique(np.concatenate(list(temp.values())[1:], axis=1), axis=1)
            self.edge_features.append(temp)

    def get_lineage(self, idx):
        return self.lineages[idx]
    
    def get_dataset(self, i):
        return CHERRYInteractionDataset(self.interactions_list[i], self.interactions_list[:min(i,2)], self.node_feature, self.edge_features[min(i,2)], multihost=self.multihost)


class CHERRYInteractionDataset(InteractionDataset):
    def __init__(self, interactions, interactions_support, node_feature, edge_feature, multihost=False):
        if len(interactions_support) == 0: # training dataset
            support_ratio = 0.7
            mask = np.zeros(len(interactions), dtype=int)
            mask[:int(len(interactions)*support_ratio)] = 1
            np.random.shuffle(mask)
            mask = mask.astype(bool)
            interactions_support = interactions[mask] # TODO: reconsider this part because we only need positive interactions for support
            interactions = interactions[~mask]
        elif len(interactions_support) >= 1: # validation or test dataset
            interactions_support = np.concatenate(interactions_support)
        super().__init__(interactions) # note: the interactions passed to super must match the edge_label_index below (because we subsample from edge_label_index using MySubsetSampler)
        
        """create graph"""
        graph_data = HomoData()
        graph_data.x = node_feature
        graph_data.edge_index = torch.from_numpy(edge_feature['edge_index']).long()

        """add edge features for infection status"""
        interactions = torch.from_numpy(interactions).long()
        #print(interactions[interactions[:,2] == 1].shape)
        interactions_support = torch.from_numpy(interactions_support).long()
        interactions_support_pos = interactions_support[interactions_support[:,2] == 1]
        # edge_label is for inspection (i.e., target edges of loss function), thus contains positive and negative edges
        graph_data.edge_label_index = interactions[:,:2].T
        graph_data.edge_label = interactions[:,2]
        # edge_index is prior knowledge (i.e, source edges when constructing GNN), thus only contains (known knowdlege and) positive edges different from edge_label_index
        graph_data.edge_index = torch.unique(torch.cat([graph_data.edge_index, interactions_support_pos[:,:2].T], dim=1), dim=1)
        
        """add deterministic predictions constructed from known edges (we also implement multi-host deterministic predictions unlike Cherry)"""
        phg2hst_target_dic = defaultdict(list)
        for hst_loc, phg_loc in interactions[:,0:2].numpy():
            phg2hst_target_dic[phg_loc].append(hst_loc)
        phg2hst_support_pos_dic = defaultdict(list)
        for hst_loc, phg_loc in interactions_support_pos[:,0:2].numpy():
            phg2hst_support_pos_dic[phg_loc].append(hst_loc)
        # predictions constructed from phage-(blastn)-known_phage-(infection)-known_host
        phg2hst_ppn_dic = self.construct_det_pred_phg_phg(phg2hst_target_dic, phg2hst_support_pos_dic, edge_feature['phage_phage_blastn'].T)
        # predictions constructed from phage-(blastp)-known_phage-(infection)-known_host
        phg2hst_ppp_dic = self.construct_det_pred_phg_phg(phg2hst_target_dic, phg2hst_support_pos_dic, edge_feature['phage_phage_blastp'].T)
        # predictions constructed from phage-(crispr)-host (note that host may be known or unknown (whereas cherry only implements known))
        phg2hst_phc_dic = self.construct_det_pred_phg_hst(phg2hst_target_dic, edge_feature['phage_host_crisprs'].T)
        # predictions constructed from phage-(blastn)-host (note that host may be known or unknown)
        phg2hst_phn_dic = self.construct_det_pred_phg_hst(phg2hst_target_dic, edge_feature['phage_host_blastn'].T)
        
        # merge the results
        phg2hst_dic = defaultdict(list)
        if not multihost: # original cherry-like prediction
            # choose single host
            for phg_loc, hst_locs in phg2hst_ppn_dic.items():
                if len(hst_locs) > 1:
                    phg2hst_ppn_dic[phg_loc] = [random.choice(hst_locs)]
            for phg_loc, hst_locs in phg2hst_phc_dic.items():
                if len(hst_locs) > 1:
                    phg2hst_phc_dic[phg_loc] = [random.choice(hst_locs)]

            # predictions constructed from one or two hops
            for phg_loc in phg2hst_target_dic.keys():
                candidates = phg2hst_ppp_dic[phg_loc] + phg2hst_phn_dic[phg_loc] + phg2hst_phc_dic[phg_loc]
                if len(set(candidates)) == 1:
                    phg2hst_dic[phg_loc].append(candidates[0])
                elif len(phg2hst_phc_dic[phg_loc]) > 0:
                    phg2hst_dic[phg_loc].append(phg2hst_phc_dic[phg_loc][0])
                elif len(phg2hst_ppp_dic[phg_loc]) > 0:
                    phg2hst_dic[phg_loc].append(phg2hst_ppp_dic[phg_loc][0])

            # predictions constructed from multi-hops (note that both of the subgraph situations in the original cherry implementation don't work due to bugs)
            num_nodes = len(graph_data.x)
            result = self.subgraph_situation_1(edge_feature['phage_phage_blastp'], num_nodes, phg2hst_phc_dic, phg2hst_target_dic)
            for phg_loc, hst_loc in result.items():
                phg2hst_dic[phg_loc] = [hst_loc] # renew
            result = self.subgraph_situation_2(edge_feature['edge_index'], num_nodes, phg2hst_target_dic)
            for phg_loc, hst_loc in result.items():
                phg2hst_dic[phg_loc] = [hst_loc] # renew

        else: # multihost mode where the original cherry implementation is slightly modified
            # predictions constructed from one or two hops
            for phg_loc in phg2hst_target_dic.keys():
                candidates = phg2hst_ppp_dic[phg_loc] + phg2hst_ppn_dic[phg_loc] + phg2hst_phn_dic[phg_loc] + phg2hst_phc_dic[phg_loc]
                phg2hst_dic[phg_loc] = list(set(candidates))
            # predictions constructed from multi-hops
            num_nodes = len(graph_data.x)
            edge_feature = np.unique(np.concatenate([edge_feature['phage_phage_blastp'], edge_feature['phage_phage_blastn']], axis=1), axis=1)
            result = self.subgraph_situation_1(edge_feature, num_nodes, phg2hst_phc_dic, phg2hst_target_dic, multihost=multihost)
            for phg_loc, hst_locs in result.items():
                phg2hst_dic[phg_loc] = list(set(phg2hst_dic[phg_loc] + hst_locs)) # add
            result = self.subgraph_situation_2(edge_feature['edge_index'], num_nodes, phg2hst_target_dic)
            for phg_loc, hst_loc in result.items():
                phg2hst_dic[phg_loc] = list(set(phg2hst_dic[phg_loc] + [hst_loc])) # add
        self.phg2hst_dic = phg2hst_dic

        """filter out unnecessary edges (for example, we don't want to use test edges to predict training interactions)"""
        interactions_all = torch.cat([interactions, interactions_support])
        loc_all = torch.unique(interactions_all[:,0:2])
        edge_index = graph_data.edge_index.T
        graph_data.edge_index = edge_index[torch.all(torch.isin(edge_index, loc_all), dim=1)].T
        self.graph_data = T.ToUndirected()(graph_data)

    def construct_det_pred_phg_phg(self, phg2hst_target_dic, phg2hst_support_pos_dic, edge_feature):
        temp = defaultdict(list)
        for phg_loc1, phg_loc2 in edge_feature:
            if len(phg2hst_target_dic[phg_loc2]) > 0:
                inspection_hosts = phg2hst_target_dic[phg_loc2]
                for hst_loc in phg2hst_support_pos_dic[phg_loc1]:
                    if hst_loc in inspection_hosts:
                        temp[phg_loc2].append(hst_loc)
            if len(phg2hst_target_dic[phg_loc1]) > 0:
                inspection_hosts = phg2hst_target_dic[phg_loc1]
                for hst_loc in phg2hst_support_pos_dic[phg_loc2]:
                    if hst_loc in inspection_hosts:
                        temp[phg_loc1].append(hst_loc)
        for k in temp:
            temp[k] = list(set(temp[k]))
        return temp

    def construct_det_pred_phg_hst(self, phg2hst_target_dic, edge_feature):
        temp = defaultdict(list)
        for phg_loc, hst_loc in edge_feature:
            if hst_loc in phg2hst_target_dic[phg_loc]:
                temp[phg_loc].append(hst_loc)
        for k in temp:
            temp[k] = list(set(temp[k]))
        return temp

    def subgraph_situation_1(self, edge_feature, num_nodes, phg2hst_phc_dic, phg2hst_target_dic, multihost=False):
        # the situation where testing phages are clustered and they have crispr host(s))
        adj = to_scipy_sparse_matrix(torch.from_numpy(edge_feature).long(), num_nodes=num_nodes)
        n_components, labels = sp.csgraph.connected_components(adj, directed=False)
        u, counts = np.unique(labels, return_counts = True)
        u = u[counts > 1]
        result = {}
        phg_locs_target = set(list(phg2hst_target_dic.keys()))
        for i in u:
            mask = labels == i
            phg_locs = np.arange(len(labels))[mask]
            if set(phg_locs) <= phg_locs_target or multihost: # if multihost mode, we ignore the first condition
                temp = []
                for phg_loc in phg_locs:
                    temp += phg2hst_phc_dic[phg_loc]
                if not multihost and len(set(temp)) == 1:
                    for phg_loc in phg_locs:
                        if temp[0] in phg2hst_target_dic[phg_loc]:
                            result[phg_loc] = temp[0]
                elif multihost and len(set(temp)) > 0:
                    for phg_loc in phg_locs:
                        hst_locs_target = set(temp) & set(phg2hst_target_dic[phg_loc])
                        if len(hst_locs_target) > 0:
                            result[phg_loc] = list(hst_locs_target)
        return result

    def subgraph_situation_2(self, edge_feature, num_nodes, phg2hst_target_dic):
        # the situation where a cluster consists of some phages and (only one) host
        hst2phg_target_dic = defaultdict(list)
        for phg_loc, hst_locs in phg2hst_target_dic.items():
            for hst_loc in hst_locs:
                hst2phg_target_dic[hst_loc].append(phg_loc)
        adj = to_scipy_sparse_matrix(torch.from_numpy(edge_feature).long(), num_nodes=num_nodes)
        n_components, labels = sp.csgraph.connected_components(adj, directed=False)
        u, counts = np.unique(labels, return_counts = True)
        u = u[counts > 1]
        result = {}
        hst_locs_target = set(list(hst2phg_target_dic.keys()))
        for i in u:
            mask = labels == i
            locs = np.arange(len(labels))[mask]
            hst_locs = set(locs) & hst_locs_target
            if len(hst_locs) == 1:
                hst_loc = list(hst_locs)[0]
                phg_locs = set(locs) & set(hst2phg_target_dic[hst_loc])
                for phg_loc in phg_locs:
                    result[phg_loc] = hst_loc
        return result
