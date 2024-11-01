import numpy as np
import glob 

def merge_fn(contig_type='species'):
    data_dir = '../data_cherry/'
    edge_index = []
    corr = []
    descriptor = []
    print(len(glob.glob(data_dir + f'pblks/{contig_type}_network*[0-9].npz')))
    print(glob.glob(data_dir + f'pblks/{contig_type}_network*[0-9].npz'))
    for fp in glob.glob(data_dir + f'pblks/{contig_type}_network*[0-9].npz'):
        data = np.load(fp)
        edge_index.append(data['edge_index'])
        corr.append(data['corr'])
        descriptor.append(data['descriptor'])

    edge_index = np.concatenate(edge_index, axis=1)
    corr = np.concatenate(corr)
    descriptor = np.concatenate(descriptor)

    np.savez(data_dir + f'pblks/{contig_type}_network.npz', edge_index=edge_index, corr=corr, descriptor=descriptor)

merge_fn('species')
#merge_fn('contigs')
print('finished')
