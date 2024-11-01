# operations common among all methods

import os
import numpy as np
import random
import torch
import wandb
import warnings
import pickle
import time
import torch.optim as optim
from evaluator import save_results
from torch.utils.data import Dataset, Sampler

# abstract runner
class Runner():
    def __init__(self, args):
        
        if len(args.model_pretrained) == 0:
            mode = 'TRN' # training
            save_path = self.create_path(args)
            model_save_path = os.path.join(args.out_dir, save_path+'.pth')
            with open(os.path.splitext(model_save_path)[0] + '-args.pkl', 'wb') as f:
                pickle.dump(args, f)
        else:
            mode = 'TST' # testing
            self.model_pretrained = args.model_pretrained
            with open(os.path.splitext(args.model_pretrained)[0] + '-args.pkl', 'rb') as f:
                args = pickle.load(f)
            save_path = self.create_path(args)
            model_save_path = os.path.join(args.out_dir, save_path+'.pth')
        
        os.makedirs(args.out_dir, exist_ok=True)
        self.model_save_path = model_save_path
        self.mode = mode
        self.args = args

        # initialize settings
        self.set_seed(args.seed)
        self.initialize_logger(save_path, args, mode)
        warnings.filterwarnings("ignore")
        
    def set_seed(self, s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms = True

    def create_path(self, args):
        path = ''
        path += 'model-{}-'.format(args.model)
        if 'data' in vars(args):
            path += 'data-{}-'.format(args.data)
        else:
            path += 'data-phd-'
        path += 'granu-{}-'.format(args.granularity)
        if args.model not in ['pblks'] and 'epoch' in vars(args):
            path += 'epoch-{}-'.format(args.epoch)
        #if 'mixup_epoch' in vars(args):
        #    path += 'mixup_s-{}-'.format(args.mixup_epoch)
        #    path += 'pu-{}-'.format(args.pu_type)
        #if 'mixup_reload' in vars(args):
        #    path += 'mixup_r-{}-'.format(args.mixup_reload)
        #if 'mixup_alpha' in vars(args):
        #    path += 'alpha-{}-'.format(args.mixup_alpha)
        #    path += 'beta-{}-'.format(args.mixup_beta)
        #if 'mixup_eps' in vars(args):
        #    path += 'eps-{}-'.format(args.mixup_eps)
        if 'aug' in vars(args):
            path += 'aug-{}-'.format(args.aug)
        if 'cl_type' in vars(args):
            path += 'cl-{}-'.format(args.cl_type)
        path += 'seed-{}-'.format(args.seed)
        for k, v in vars(args).items():
            if args.model in k:
                path += '{}-{}-'.format(k[len(args.model)+1:], v)
        return path[:-1]

    def initialize_logger(self, name, args, mode):
        wandb.login()
        # initialize config of wandb
        wandb.init(
            project = 'phage-host-interaction-p1',
            name = mode + '-' + name,
            config = vars(args) | {'mode': mode},
            #entity = 'yohei', # your username or team name where you're sending runs
        )
        return

    def start_timer_dataloading(self):
        print(" |- Start preparing dataset...")
        self.start_dataload = time.time()

    def end_timer_dataloading(self):
        self.used_dataload = time.time() - self.start_dataload
        print("     |- loading [ok].")
        print("     |- used time:", round(self.used_dataload,2), "s")

    def start_timer_running(self):
        print(" |- Start {} mode...".format(self.mode))
        self.start_running = time.time()

    def end_timer_running(self):
        used_running = time.time() - self.start_running
        print("     |- running [ok].")
        print("     |- used time:", round(used_running,2), "s")
        print(" |- Total time:", round(used_running+self.used_dataload,2))

    def test(self):
        """evaluation starts"""
        results = {}
        with torch.no_grad():
            for i in range(2, len(self.dataset_stat)):
                results_tst = self.one_epoch(1, i, self.dataloaders[i], query=self.dataset_stat[i]['query'])
                results.update({self.dataset_stat[i]['name']+'_'+k:v for k,v in results_tst.items()})
        dic = save_results(results, self)
        wandb.log(dic)
        return

    def train(self, lr=1e-3):
        """training starts"""
        print(f" |- Total number of parameters is %d" %(sum([p.nelement() for p in self.model.parameters()])))
        print("  |- Training started ...")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        n_epochs = self.args.epoch
        n_digits = len(str(n_epochs+1))
        bad_counts = 0
        bad_limit = -1
        bst_val_score = 0
        bst_model = self.model
        for epoch in range(1, n_epochs+1):

            trn_results = self.one_epoch(epoch, 0, self.dataloaders[0], query=self.dataset_stat[0]['query'])
            epoch_info = f'epoch = {{:0={n_digits}}} , trn loss = {{:.6f}}'.format(epoch, trn_results['loss'])
            results = {'trn_'+k:v for k,v in trn_results.items()}

            with torch.no_grad():
                val_results = self.one_epoch(epoch, 1, self.dataloaders[1], query=self.dataset_stat[1]['query'])
                epoch_info += ' , val loss = {:.6f}'.format(val_results['loss'])
                results.update({'val_'+k:v for k,v in val_results.items()})

            print(epoch_info)

            wandb.log({k:v for k,v in results.items() if isinstance(v, float)})

            criterion = 'val_PR-AUC_hst-species-phg-species'
            if criterion in results and results[criterion] > bst_val_score:
                bst_val_score = results[criterion]
                bad_counts = 0
                torch.save(self.model.state_dict(), self.model_save_path)
            else:
                bad_counts += 1
            if bad_limit > 0 and bad_counts >= bad_limit:
                break
        return
