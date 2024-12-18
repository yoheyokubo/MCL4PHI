import os
import argparse

def all_args():
    parser = argparse.ArgumentParser(description='<Computational Tools for Phage-Host Interaction Prediction>')
    
    # common
    parser.add_argument('--model_pretrained', default='', type=str, required=False, help='the path for a pretrained model')
    parser.add_argument('--model', default="cl4phi", type=str, required=False, choices=['pblks', 'cl4phi', 'cherry'], help='prediction model type')
    parser.add_argument('--out_dir', type=str, required=True, help="directory for saving a trained model and results")
    parser.add_argument('--data_dir', type=str, required=True, help="directory for loading data")
    parser.add_argument('--data', type=str, required=False, default='cherry', choices=['cherry'], help="dataset type")
    parser.add_argument('--granularity', type=str, required=False, default='contigs', choices=['contigs', 'species'], help='input granurarity fed into the model')
    parser.add_argument('--seed', default=123, type=int, required=False, help='seed for repetition')
    parser.add_argument('--epoch', default=150, type=int, required=False, help='number of epochs in training deep learning models')
    parser.add_argument('--aug', default='0.0', type=str, required=False, help='whether or not use augmentation')

    # cherry
    parser.add_argument('--cherry_multihost', action='store_true', help='settings for the deterministic prediction of cherry (the default is single-host)')
    parser.add_argument('--cherry_period', default=1, type=int, required=False, help='epoch period for reloading training dataloader')

    args = parser.parse_args()
    
    return args

args = all_args()
