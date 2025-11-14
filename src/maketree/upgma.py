import sys
sys.path.append('./clone/rl_leiden')

from src.maketree.node import Node, get_center
from src.distance import l1_distance, l2_distance

from scipy.stats import mode
import numpy as np
from tqdm import tqdm

from src.utils import get_root_data

def maketree_UPGMA(cnv, labels, dist_func=l1_distance):
    """
    UPGMA 
    """
    nodes = [Node(v=get_center(cnv[np.where(labels == label)]),
             n=(labels == label).sum(),  
             label=label) for label in np.unique(labels)]

    r_v, r_l = get_root_data(cnv)
    root = Node(v=r_v, n=cnv.shape[0] // 10, label=r_l)  # virtual root

    n = len(nodes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i, j] = dist_func(nodes[i].v, nodes[j].v)
            dist_matrix[j, i] = dist_matrix[i, j]

    unconnected_nodes = set(range(n))

    if unconnected_nodes:
        root_dists = [dist_func(root.v, nodes[i].v) for i in unconnected_nodes]
        nearest_to_root_idx = list(unconnected_nodes)[np.argmin(root_dists)]
        root.add_offspring(nodes[nearest_to_root_idx])
        nodes[nearest_to_root_idx].set_parent(root)
        unconnected_nodes.remove(nearest_to_root_idx)
    
    with tqdm(total=len(nodes)-1) as pbar:
        pbar.set_description('Making UPGMA Tree')
        
        while unconnected_nodes:
            min_dist = float('inf')
            best_connected = -1
            best_unconnected = -1
            
            connected_nodes = []
            for i in range(n):
                if i not in unconnected_nodes and nodes[i].parent is not None:
                    connected_nodes.append(i)
            
            for connected_idx in connected_nodes:
                for unconnected_idx in unconnected_nodes:
                    dist = dist_matrix[connected_idx, unconnected_idx]
                    if dist < min_dist:
                        min_dist = dist
                        best_connected = connected_idx
                        best_unconnected = unconnected_idx
            
            if best_connected != -1 and best_unconnected != -1:
                nodes[best_connected].add_offspring(nodes[best_unconnected])
                nodes[best_unconnected].set_parent(nodes[best_connected])
                unconnected_nodes.remove(best_unconnected)
                pbar.update(1)
            else:
                break
    
    return root