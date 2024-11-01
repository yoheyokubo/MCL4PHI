import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, precision_recall_curve, roc_curve
import torch

def lineage_distance(lin1, lin2): # assume None has been replaced by negative int
    d = 2 * torch.argmax(((lin1 >= 0) & (lin2 >= 0) * ((lin1 - lin2) == 0)).to(torch.int32), axis=-1)
    return d

class PrecisionRecallEvaluator:
    def __init__(self, granularity):
        self.granularity = granularity

    def get_results(self, y_true, y_pred, y_prob, name_hst='species', name_phg='species'):
        results = {}
        name = 'hst-{}-phg-{}'.format(name_hst, name_phg)
        results['PR-AUC_'+name] = average_precision_score(y_true, y_prob)
        results['ROC-AUC_'+name] = roc_auc_score(y_true, y_prob)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1])
        results['precision_'+name] = p[0]
        results['recall_'+name] = r[0]
        results['f1score_'+name] = f1[0]
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        results['fpr_'+name] = fpr
        results['tpr_'+name] = tpr
        results['thresholds_'] = thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        results['pre_'+name] = precision
        results['rec_'+name] = recall
        results['thr_'+name] = thresholds
        return results
    
    def merge_taxa(self, y_true, y_pred, y_prob, lins, key=[0]): # key is a list ([0], [1], [0,1]) to merge predictions sharing the same taxa along (where 0: host, 1: phage)
        ranks = ['species', 'genus', 'family']
        lin_old = lins[:,:,0]
        names = ['name_hst', 'name_phg']
        results = {}
        for r in range(1, len(ranks)):
            lin_new = np.copy(lin_old)
            lin_new[:,key] = lins[:,key,r] # update with key's (new) taxa at rank r
            mask_invalid = lin_new < 0
            lin_new[mask_invalid] = lin_old[mask_invalid] # keep key's old taxa if the new taxa is invalid (i.e., taxid information is missing)
            u, indices = np.unique(lin_new, axis=0, return_inverse=True)
            y_true_temp, y_pred_temp, y_prob_temp = [], [], []
            for i in range(len(u)):
                mask = indices == i # mask to extract all the predictions with key and query from the same taxa
                y_true_temp.append(np.max(y_true[mask]))
                y_pred_temp.append(np.max(y_pred[mask]))
                y_prob_temp.append(np.max(y_prob[mask]))
            y_true_r = np.array(y_true_temp, dtype=y_true.dtype)
            y_pred_r = np.array(y_pred_temp, dtype=y_pred.dtype)
            y_prob_r = np.array(y_prob_temp, dtype=y_prob.dtype)
            kwargs = {names[i]: ranks[r] if i in key else ranks[0] for i in range(2)} # e.g., name_hst: 'species', name_phg: 'genus' if key == [1] and r == 1
            results_r = self.get_results(y_true_r, y_pred_r, y_prob_r, **kwargs)
            results.update(results_r)
            lin_old = lin_new
        
        return results

    def evaluate(self, y_true, y_pred, y_prob, lins, query='none'): # lins: ndarray for lineage information of shape (number of predictions, 2, 9), where 2 represents host and phage
        if self.granularity == 'contigs': # merge interactions from contig-level to species-level
            u, indices = np.unique(lins[:,:,0], axis=0, return_inverse=True) # assume that there is no contigs with missing species-level (or distinct placeholder) taxid
            y_true_temp, y_pred_temp, y_prob_temp, lins_temp = [], [], [], []
            for i in range(len(u)):
                mask = indices == i
                y_true_temp.append(np.max(y_true[mask])) # one of the contigs within the taxa has the 'infection' status then so does the taxa
                y_pred_temp.append(np.max(y_pred[mask]))
                y_prob_temp.append(np.max(y_prob[mask]))
                lins_temp.append(u[i])
            y_true = np.array(y_true_temp, dtype=y_true.dtype)
            y_pred = np.array(y_pred_temp, dtype=y_true.dtype)
            y_prob = np.array(y_prob_temp, dtype=y_true.dtype)
            lins = np.stack(lins_temp)
        results = self.get_results(y_true, y_pred, y_prob)
        ranks = ['species', 'genus', 'family']
        if query == 'phage': # evaluation merged w.r.t. host taxa
            results.update(self.merge_taxa(y_true, y_pred, y_prob, lins, key=[0]))
        elif query == 'host': # evaluation merged w.r.t. phage taxa
            results.update(self.merge_taxa(y_true, y_pred, y_prob, lins, key=[1]))
        elif query == 'pair': # evaluation merged w.r.t. phage and host taxa
            results.update(self.merge_taxa(y_true, y_pred, y_prob, lins, key=[0,1]))
        return results

def save_results(results, runner):
    dic = {k: v for k, v in results.items() if isinstance(v, float)}
    dic.update(seed=runner.args.seed, model=runner.args.model)
    df = pd.DataFrame.from_dict(dic, orient="index")
    mode = runner.mode
    df.to_csv(os.path.splitext(runner.model_save_path)[0]+f"-{mode}data.csv")
    dic_array = {k:v.tolist() for k, v in results.items() if not isinstance(v, float)}
    dic_array.update(seed=runner.args.seed, model=runner.args.model)
    with open(os.path.splitext(runner.model_save_path)[0]+f"-{mode}data.json", "w") as f:
        json.dump(dic_array, f, indent=2)
    return dic # return for wandb.log
