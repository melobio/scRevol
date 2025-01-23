import pandas as pd
import numpy as np
import random
import torch
import os
import networkx as nx
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import precision_recall_curve, auc
from itertools import product
from tqdm import tqdm
from scipy.stats import mode


def set_seed_everywhere(seed=0):
    np.random.seed(seed)
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def load_data(data_dir, dataset_id):
    for f in os.listdir(f'{data_dir}'):
        f_id = eval(f.split('_')[0])
        if f_id == dataset_id:
            dataset_name = f
            break
    else:
        print(f"No file found starts with id {dataset_id}, please check.")
        quit()

    data = pd.read_csv(os.path.join(data_dir, dataset_name, 'matrix.tsv'), sep="\t", index_col=0)
    c2cl = pd.read_csv(os.path.join(data_dir, dataset_name, 'cell2clone.tsv'), sep="\t")
    cl2idx = {x:i for i, x in enumerate(set(c2cl.clone))}
    c2cl['idx'] = c2cl.clone.map(cl2idx)

    return data, c2cl, dataset_name

def load_real_data(data_dir, dataset_name, remove_no_epi=False):
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

''' metrics '''
def cal1B(truth, pred):
    return (truth + 1 - min(truth+1, abs(pred-truth))) / float(truth+1)

def cal2(truth, pred):
    '''
     from nature: mean of aupr and ajsd
        ajsd: mean JS divergence of rows between real and pred CCM
    '''
    real_ccm = CCM(truth)
    pred_ccm = CCM(pred)
    # aupr = compute_aupr(real_ccm, pred_ccm)
    aupr = 0
    ajsd = np.array(jensenshannon(real_ccm, pred_ccm, axis=1) ** 2).mean()
    return (aupr + ajsd) / 2 

def cal3(truth, pred, cnv):
    '''
        from nature: Pearson correlation coefficient between [CCM, ADM, ADM.T, CM]s
    '''
    real_root = maketree(cnv, truth)
    real_ccm = CCM(truth)
    real_adm = ADM(cnv, truth, real_root)
    real_cm  = CM(real_ccm, real_adm)
    real_matrix = np.concatenate([real_ccm, real_adm, real_adm.T, real_cm]).flatten()

    pred_root = maketree(cnv, pred)
    pred_ccm = CCM(pred)
    pred_adm = ADM(cnv, pred, pred_root)
    pred_cm  = CM(pred_ccm, pred_adm)
    pred_matrix = np.concatenate([pred_ccm, pred_adm, pred_adm.T, pred_cm]).flatten()

    # pcc = np.mean((real_matrix - real_matrix.mean()) * (pred_matrix - pred_matrix.mean())) / (real_matrix.std() * pred_matrix.std())
    # or simplified as 
    pcc = np.corrcoef(real_matrix, pred_matrix)[0, 1]
    return pcc

''' matrix calculation '''
def CCM(label):
    if isinstance(label, list):
        label = np.array(label)
    num_samples = label.shape[0]
    unique_clusters = np.unique(label)
    ccm_matrix = np.zeros((num_samples, num_samples), dtype=int)

    for cluster_id in unique_clusters:
        cluster_indices = np.where(label == cluster_id)[0]
        ccm_matrix[np.ix_(cluster_indices, cluster_indices)] = 1
    
    return ccm_matrix

def CM(ccm, adm):
    return 1 - ccm - adm - adm.T

def ADM(cnv, labels, root):
    m = len(cnv) 
    ADM = np.zeros((m, m), dtype=int)

    nodes = get_treenodes(root)
    label_to_node = build_label_to_node_map(nodes)

    cell_to_ancestors = {i:[] for i in range(m)}
    for i in range(m):
        cell_cluster_label = labels[i]
        cell_node = label_to_node[cell_cluster_label]
        ancestors = get_ancestors(cell_node)
        cell_to_ancestors[i].extend([a.label for a in ancestors])

    for i in range(m):
        for j in range(m):
            if labels[j] in cell_to_ancestors[i]:
                ADM[j, i] = 1

    return ADM

def compute_aupr(matrix1, matrix2):
    y_true = matrix1.flatten()
    y_scores = matrix2.flatten()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    aupr = auc(recall, precision)
    
    return aupr

''' tree construction '''
def get_ancestors(node):
    ancestors = []
    while node.parent is not None:
        ancestors.append(node.parent)
        node = node.parent
    return ancestors

def build_label_to_node_map(nodes):
    label_to_node = {}
    for node in nodes:
        label_to_node[node.label] = node
    return label_to_node
  
class Node(object):
    def __init__(self, v, n, label, parent=None, offsprings=None):
        self.v = v
        self.n = n
        self.label = label
        self.parent = parent
        if offsprings is not None:
            self.offsprings = offsprings
        else:
            self.offsprings = []
    
    def set_parent(self, parent):
        self.parent = parent
        
    def set_offsprings(self, offsprings):
        self.offsprings = offsprings
        
    def add_offspring(self, offspring):
        self.offsprings.append(offspring)

def l1_distance(a, b):
    return np.abs(a-b).sum()

def l2_distance(a, b):
    return np.square(a - b).mean()

def get_center(cnv):
    # return np.mean(cnv, axis=0) # use mean as center
    most_vec = mode(cnv, axis=0).mode.flatten()
    return np.array(most_vec)
    
def   maketree(cnv, labels, dist_func=l1_distance):
    return maketree_MST(cnv, labels.flatten(), dist_func)

def maketree_MST(cnv, labels, dist_func=l1_distance):
    nodes = [Node(v=get_center(cnv[np.where(labels == label)]),n=(labels == label).sum(),  label=label) for label in np.unique(labels)]
    r_v, r_l = get_root_data(cnv)
    root = Node(v=r_v, n=cnv.shape[0] // 10, label=r_l)
    nodes.append(root) # root由虚拟方法构建

    nodes_used = [root]
    nodes_unused = [node for node in nodes if node not in nodes_used]
    edges = []

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i < j:
                dist = dist_func(node1.v, node2.v)
                edges.append((dist, node1, node2))

    edges.sort(key=lambda x: x[0])

    while len(nodes_used) < len(nodes):
        for dist, node1, node2 in edges:
            # if root in [node1, node2] and len(root.offsprings) > 0 :
            #     continue # root have only one child -> real root
            if (node1 in nodes_used and node2 not in nodes_used):
                node1.add_offspring(node2)
                node2.set_parent(node1)
                nodes_used.append(node2)
                nodes_unused.remove(node2)
                break
            elif (node2 in nodes_used and node1 not in nodes_used):
                node2.add_offspring(node1)
                node1.set_parent(node2)
                nodes_used.append(node1)
                nodes_unused.remove(node1)
                break
    # root have only one child -> real root
    # assert len(root.offsprings) == 1
    # root_real = root.offsprings[0]
    # root_real.set_parent(None)
    # return root_real

    for root_real in root.offsprings:
        root_real.set_parent(None)

    return root.offsprings

def merge_nodes(nodes):
    assert sum([0 if node.parent is None else 1 for node in nodes]) == 0 # 只有根能合并
    new_node = Node(v=0, n=0, label=nodes[0].label)
    for node in nodes:
        new_node.n += node.n
        for offspring in node.offsprings:
            new_node.add_offspring(offspring)
            offspring.set_parent(new_node)
    return new_node

def showtree(root):
    for child in root.offsprings:
        print(root.label, child.label)
        showtree(child)

def get_parent_child_pairs(root):
    parent_child_pairs = []
    
    def traverse(node):
        for child in node.offsprings:
            parent_child_pairs.append([node.label, child.label])
            traverse(child)

    traverse(root)
    
    return np.array(parent_child_pairs)

def drawtree(tree_path, path=None):
    df = tree_path
    G = nx.DiGraph()
    G.add_edges_from(zip(df['parent'], df['son']))

    # Draw the tree
    plt.figure(figsize=(10, 8))
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # Tree layout
    nx.draw(G, pos, with_labels=True, arrows=False, node_size=1500, node_color="lightblue", font_size=10, font_weight="bold")
    plt.title("Tree Structure", fontsize=14)
    plt.savefig(path)
    

def get_treenodes(root):
    nodes = []
    nodes_haschild = [root]
    while len(nodes_haschild) > 0:
        node = nodes_haschild.pop()
        nodes_haschild.extend(node.offsprings)
        nodes.append(node)
    return nodes

def calculate_bounds(reference):
    down = np.mean(reference.mean(axis=1)) - 2 * np.mean(reference.std(axis=1))
    up = np.mean(reference.mean(axis=1)) + 2 * np.mean(reference.std(axis=1))
    return down, up

def classify_cnv(data, down, up):
    cnv_state = np.where(data < down, 1,  # loss of one copy
                         np.where(data < up, 2,  # neutral
                                  3))  # gain of one copy
    return cnv_state



import numpy as np
from scipy.stats import mode

def get_root_data(cnv):
    root_data = mode(cnv, axis=0).mode[0] # 获取特征的众数组成虚拟的根
    return root_data, -1 # 虚拟根的标签为-1


def get_founder(root, nodes, dist_func):
    unused = [node for node in nodes if node is not root]
    dists = [dist_func(root.v, node.v) for node in unused]
    return unused[np.argmin(dists)]

def get_node(root_label, nodes):
    for node in nodes:
        if node.label == root_label:
            return node