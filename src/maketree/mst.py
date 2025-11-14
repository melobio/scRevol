import sys
sys.path.append('./clone/rl_leiden')

from src.maketree.node import Node, get_center
from src.distance import l1_distance, l2_distance

from scipy.stats import mode
import numpy as np
from tqdm import tqdm

from src.utils import get_root_data

def maketree_MST(cnv, labels, dist_func=l1_distance):
    nodes = [Node(v=get_center(cnv[np.where(labels == label)]),
                  n=(labels == label).sum(),  
                  label=label) for label in np.unique(labels)]
    r_v, r_l = get_root_data(cnv)
    root = Node(v=r_v, n=cnv.shape[0] // 10, label=r_l)
    nodes.append(root) # visual root

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
            if root in [node1, node2] and len(root.offsprings) > 0 :
                continue # root have only one child -> real root
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
    # return root
    # root have only one child -> real root
    assert len(root.offsprings) == 1
    root_real = root.offsprings[0]
    root_real.set_parent(None)
    return root_real