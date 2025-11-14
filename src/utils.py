import pandas as pd
import numpy as np
import random
import torch
import os

from scipy.stats import mode


def set_seed_everywhere(seed=0):
    np.random.seed(seed)
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

def id2name(data_dir, dataset_id):
    for f in os.listdir(f'{data_dir}'):
        f_id = eval(f.split('_')[0])
        if f_id == dataset_id:
            dataset_name = f
            break
    else:
        print(f"No file found starts with id {dataset_id}, please check.")
        quit()
    
    return dataset_name

def load_data(data_dir, dataset_id):
    dataset_name = id2name(data_dir, dataset_id)
    # NOTE: Adjust the following file names according to your actual
    # simulated data layout. Here we assume "matrix.tsv", "cell2clone.tsv",
    # and "tree.tsv" inside each <dataset_id> directory.
    data = pd.read_csv(os.path.join(data_dir, dataset_name, 'matrix.tsv'), sep="\t", index_col=0)
    c2cl = pd.read_csv(os.path.join(data_dir, dataset_name, 'cell2clone.tsv'), sep="\t")
    tree = pd.read_csv(os.path.join(data_dir, dataset_name, 'tree.tsv'), sep='\t')

    labels = set(tree.values.reshape(-1).tolist())
    cl2idx = {x:i for i, x in enumerate(set(labels))}

    c2cl['idx'] = c2cl.clone.map(cl2idx)

    tree.parent = tree.parent.map(cl2idx)
    tree.son = tree.son.map(cl2idx)

    return data, c2cl, dataset_name, tree, cl2idx

def load_real_data(data_dir, dataset_name, remove_no_epi=False):
    # NOTE: Adjust the expression file name pattern if needed.
    # Here we assume two files named "<dataset_name>.csv" and "{meta_name}.csv".
    data = pd.read_csv(os.path.join(data_dir, f'{dataset_name}.csv'), index_col=0).T
    data.index = data.index.str.strip()
    meta_name = dataset_name.split('_')[0] + '_meta'
    meta_path = os.path.join(data_dir, f'{meta_name}.csv')
    meta = None
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path, index_col=0)
    if remove_no_epi and meta is not None:
        meta.columns = [col.strip() for col in meta.columns]
        non_epis = meta[meta['celltype'] == 'non-Epi'].index
        data.drop(index=non_epis, inplace=True)
    return data, None, dataset_name, meta


def calculate_bounds(reference):
    down = np.mean(reference.mean(axis=1)) - 2 * np.mean(reference.std(axis=1))
    up = np.mean(reference.mean(axis=1)) + 2 * np.mean(reference.std(axis=1))
    return down, up

def classify_cnv(data, down, up):
    cnv_state = np.where(data < down, 1,  # loss of one copy
                         np.where(data < up, 2,  # neutral
                                  3))  # gain of one copy
    return cnv_state

def get_root_data(cnv):
    root_data = mode(cnv, axis=0, keepdims=True).mode[0] # 获取特征的众数组成虚拟的根
    return root_data, -1 # 虚拟根的标签为-1


def get_founder(root, nodes, dist_func):
    unused = [node for node in nodes if node is not root]
    dists = [dist_func(root.v, node.v) for node in unused]
    return unused[np.argmin(dists)]

def get_node(root_label, nodes):
    for node in nodes:
        if node.label == root_label:
            return node