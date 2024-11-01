# code reference: https://github.com/KennthShang/CHERRY/blob/main/utils.py

import subprocess

def make_protein_clusters_mcl(abc_fp, out_p, inflation=2):
    print("Running MCL...")
    abc_fn = "merged"
    mci_fn = '{}.mci'.format(abc_fn)
    mci_fp = out_p + mci_fn
    mcxload_fn = '{}_mcxload.tab'.format(abc_fn)
    mcxload_fp = out_p + mcxload_fn
    # make the input undirected and scaled by neg-log-10 with maximum 200
    subprocess.check_call("mcxload -abc {0} --stream-mirror --stream-neg-log10 -stream-tf 'ceil(200)' -o {1}"
                          " -write-tab {2}".format(abc_fp, mci_fp, mcxload_fp), shell=True)
    mcl_clstr_fn = "{0}_mcl{1}.clusters".format(abc_fn, int(inflation*10))
    mcl_clstr_fp = out_p + mcl_clstr_fn
    subprocess.check_call("mcl {0} -I {1} -use-tab {2} -o {3}".format(
        mci_fp, inflation, mcxload_fp, mcl_clstr_fp), shell=True)
    return mcl_clstr_fp


