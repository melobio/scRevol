import sys
sys.path.append('./clone/rl_leiden')

from src.maketree.node import Node, get_center
from src.distance import l1_distance, l2_distance

import numpy as np
from tqdm import tqdm

from src.utils import get_root_data

def maketree_NJ(cnv, labels, dist_func=l1_distance):
    """
    Neighbor Joining
    """
    nodes = [Node(v=get_center(cnv[np.where(labels == label)]),
             n=(labels == label).sum(),
             label=label) for label in np.unique(labels)]
    
    n = len(nodes)
    r_v, r_l = get_root_data(cnv)
    root = Node(v=r_v, n=cnv.shape[0] // 10, label=r_l)  # virtual root

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = dist_func(nodes[i].v, nodes[j].v)
            D[i, j] = dist
            D[j, i] = dist

    unconnected_indices = set(range(n))

    subtree_roots = set()
    
    with tqdm(total=n-1) as pbar:
        pbar.set_description('Making Neighbor Joining Tree')
        
        while len(unconnected_indices) > 1:
            current_indices = list(unconnected_indices)
            current_n = len(current_indices)
            
            if current_n < 2:
                break

            r = np.zeros(current_n)
            for i in range(current_n):
                r[i] = np.sum([D[current_indices[i], current_indices[j]] 
                              for j in range(current_n)]) / (current_n - 2)

            Q = np.zeros((current_n, current_n))
            for i in range(current_n):
                for j in range(i+1, current_n):
                    idx_i = current_indices[i]
                    idx_j = current_indices[j]
                    Q[i, j] = (current_n - 2) * D[idx_i, idx_j] - r[i] - r[j]
                    Q[j, i] = Q[i, j]

            min_q = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(current_n):
                for j in range(i+1, current_n):
                    if Q[i, j] < min_q:
                        min_q = Q[i, j]
                        merge_i, merge_j = i, j
            
            if merge_i == -1 or merge_j == -1:
                break

            idx_i = current_indices[merge_i]
            idx_j = current_indices[merge_j]

            dist_ij = D[idx_i, idx_j]
            dist_ui = 0.5 * dist_ij + (r[merge_i] - r[merge_j]) / (2 * (current_n - 2))
            dist_uj = dist_ij - dist_ui

            nodes[idx_i].branch_length = dist_ui
            nodes[idx_j].branch_length = dist_uj

            avg_dist_i = np.mean([D[idx_i, k] for k in current_indices if k != idx_i])
            avg_dist_j = np.mean([D[idx_j, k] for k in current_indices if k != idx_j])
            
            if avg_dist_i <= avg_dist_j:
                parent_node = nodes[idx_i]
                child_node = nodes[idx_j]
            else:
                parent_node = nodes[idx_j]
                child_node = nodes[idx_i]

            parent_node.add_offspring(child_node)
            child_node.set_parent(parent_node)

            unconnected_indices.remove(idx_i)
            unconnected_indices.remove(idx_j)
            subtree_roots.add(current_indices[merge_i] if parent_node == nodes[idx_i] else current_indices[merge_j])
            
            pbar.update(1)

    if len(unconnected_indices) == 1:
        last_node_idx = list(unconnected_indices)[0]

        root.add_offspring(nodes[last_node_idx])
        nodes[last_node_idx].set_parent(root)

    for subtree_root_idx in subtree_roots:
        subtree_root = nodes[subtree_root_idx]
        if subtree_root.parent is None:  
            root.add_offspring(subtree_root)
            subtree_root.set_parent(root)
    
    return root