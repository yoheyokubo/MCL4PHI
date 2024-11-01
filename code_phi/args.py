import os
import argparse

def all_args():
    parser = argparse.ArgumentParser(description='<Computational Tools for Phage-Host Interaction Prediction>')
    
    # common
    parser.add_argument('--model_pretrained', default='', type=str, required=False, help='the path for a pretrained model')
    parser.add_argument('--model', default="cl4phi", type=str, required=False, choices=['pblks', 'cl4phi', 'cherry', 'phagetb', 'graph', 'gumbel'], help='prediction model type')
    parser.add_argument('--out_dir', type=str, required=True, help="directory for saving a trained model and results")
    parser.add_argument('--data_dir', type=str, required=True, help="directory for loading data")
    parser.add_argument('--data', type=str, required=False, default='phd', choices=['phd', 'cherry'], help="dataset type")
    parser.add_argument('--granularity', type=str, required=False, default='contigs', choices=['contigs', 'species'], help='input granurarity fed into the model')
    parser.add_argument('--seed', default=123, type=int, required=False, help='seed for repetition')
    parser.add_argument('--epoch', default=150, type=int, required=False, help='number of epochs in training deep learning models')
    #parser.add_argument('--mixup_epoch', default=0, type=int, required=False, help='number of epochs in training where starts mixup for pu learning')
    #parser.add_argument('--pu_type', default='0', type=str, required=False, help='strategy for pu learning')
    #parser.add_argument('--mixup_reload', default=5, type=int, required=False, help='interval in reloading mixuped samples for pu learning')
    #parser.add_argument('--mixup_alpha', default=1, type=int, required=False, help='alpha in beta distribution')
    #parser.add_argument('--mixup_beta', default=1, type=int, required=False, help='beta in beta distribution')
    #parser.add_argument('--mixup_eps', default=0, type=float, required=False, help='eps for adjusting dicision boundary')
    parser.add_argument('--aug', default='0.0', type=str, required=False, help='whether or not use augmentation')
    parser.add_argument('--cl_type', default='1', type=str, required=False, help='type for contrastive learning')

    # cherry
    parser.add_argument('--cherry_multihost', action='store_true', help='settings for the deterministic prediction of cherry (the default is single-host)')
    parser.add_argument('--cherry_period', default=1, type=int, required=False, help='epoch period for reloading training dataloader')

    args = parser.parse_args()
    
    return args

args = all_args()
