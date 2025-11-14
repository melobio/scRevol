import numpy as np

from src.maketree.upgma import maketree_UPGMA
from src.maketree.mst import maketree_MST
from src.maketree.neighbor_join import maketree_NJ
from src.distance import l1_distance, l2_distance

def maketree(cnv, labels, dist_func=l1_distance, method='mst'):
    if method == 'upgma':
        return maketree_UPGMA(cnv, labels.flatten(), dist_func)
    elif method == 'mst':   
        return maketree_MST(cnv, labels.flatten(), dist_func)
    elif method == 'nj':    
        return maketree_NJ(cnv, labels.flatten(), dist_func)
    else:
        raise ValueError(f"Unknown method {method} for tree construction.")
    
from src.maketree.utils import get_ancestors, build_label_to_node_map, get_treenodes, get_parent_child_pairs