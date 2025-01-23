import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import argparse

from net import *
from utils import *
from cluster import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='CNV_multiSample_martix')
    parser.add_argument('--data_name', default='data2filtered_CNV')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    data_dir = f'/home/ubuntu/duxinghao/clone/data/{args.data_type}'
    ds = args.data_name
    log_dir = f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{args.data_type}/rl_leiden/CNV/leiden/{ds}'

    data, c2cl, dataset_name, non_epis = load_real_data(data_dir, ds, remove_no_epi=False)
    df = pd.read_csv(f'{log_dir}/tree_path.csv')
    c2cl =  pd.read_csv(f'{log_dir}/cell2cluster.csv', index_col=0)
    pred_labels = c2cl.values.flatten()

    roots = maketree(cnv=data.values, labels=pred_labels, dist_func=l2_distance)
    roots_labels = [r.label for r in roots]
    print(f'root offsprings: {roots_labels}')
    root = merge_nodes(roots)
    for label in roots_labels:
        c2cl['cluster'].replace(label, root.label, inplace=True)

    # showtree(root)
    tree_df = pd.DataFrame(data=get_parent_child_pairs(root), columns=['parent', 'son'])
    # tree_df.to_csv(os.path.join(log_dir, 'tree_path.csv'), index=None)

    outdir = f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{args.data_type}/rl_leiden/CNV/leiden/{ds}_new'
    os.makedirs(outdir, exist_ok=True)
    c2cl.to_csv(os.path.join(outdir, 'c2cl_new.csv'))
    drawtree(tree_df, os.path.join(outdir, 'tree_new.pdf'))

    # os.remove(f'{dir}/tree.png')

