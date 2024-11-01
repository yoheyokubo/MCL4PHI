import json
import numpy as np
import scipy as sp

modes = ['trn', 'val', 'tst']
types = ['contigs', 'species']
out_fn = '../data_cherry/cherry/'
# phage-host
for mode in modes:
    for t in types:
        for method in ['blastn', 'crisprs']:
            with open(out_fn + f'phage_host/{method}_{mode}_{t}_phages.json') as f:
                row_names = np.array(json.load(f))
            with open(out_fn + f'phage_host/{method}_{mode}_{t}_hosts.json') as f:
                col_names = np.array(json.load(f))
            mat = sp.sparse.load_npz(out_fn + f'phage_host/{method}_{mode}_{t}_mat.npz')
            mat = mat.tocsr()
            row, col = mat.nonzero()
            edge_index = np.stack([row_names[row], col_names[col]])
            edge_attr = mat[row, col]
            np.savez(out_fn + f'phage_host_{method}_{mode}_{t}.npz', edge_index=edge_index, edge_attr=edge_attr)

# phage-phage
for mode in modes:
    for t in types:
        with open(out_fn + f'phage_phage/proteins_{mode}_{t}.json') as f:
            names = json.load(f)
        mat = sp.sparse.load_npz(out_fn + f'phage_phage/proteins_{mode}_{t}_mat.npz')
        mat = mat.tocsr()
        row_names = col_names = np.array(names)
        row, col = mat.nonzero()
        edge_index = np.stack([row_names[row], col_names[col]])
        edge_attr = mat[row, col]
        np.savez(out_fn + f'phage_phage_blastp_{mode}_{t}.npz', edge_index=edge_index, edge_attr=edge_attr)

for mode in modes:
    for t in types:
        with open(out_fn + f'phage_phage_identity/blastn_{mode}_{t}_phages.json') as f:
            names = json.load(f)
        mat = sp.sparse.load_npz(out_fn + f'phage_phage_identity/blastn_{mode}_{t}_mat.npz')
        mat = mat.tocsr()
        row_names = col_names = np.array(names)
        row, col = mat.nonzero()
        edge_index = np.stack([row_names[row], col_names[col]])
        edge_attr = mat[row, col]
        np.savez(out_fn + f'phage_phage_blastn_{mode}_{t}.npz', edge_index=edge_index, edge_attr=edge_attr)

print('finished')
