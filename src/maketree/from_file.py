import os
import numpy as np
import pandas as pd

import sys
sys.path.append('./clone/rl_leiden')
from src.maketree.node import Node
from src.maketree.utils import showtree


def get_node_from_name(nodes, name):
    candidates = [node for node in nodes if node.label==name]
    assert len(candidates) == 1
    return candidates[0]

def maketree_from_file(tree):
    labels = set(tree.values.reshape(-1).tolist())
    nodes = [Node(v=None, n=0, label=label) for label in labels]

    root = get_node_from_name(nodes, tree.iloc[0,0])
    root.set_parent(None)

    for parent_name, child_name in tree.values:
        parent = get_node_from_name(nodes, parent_name)
        child = get_node_from_name(nodes, child_name)
        parent.add_offspring(child)
        child.set_parent(parent) 

    return root


if __name__ == '__main__':
    data_path = './clone/data/data_large2/1_clone5_error0.25'
    tree = pd.read_csv(os.path.join(data_path, 'tree.tsv'), sep='\t')
    c2cl = pd.read_csv(os.path.join(data_path, 'cell2clone.tsv'), sep='\t')
    cl2idx = {x:i for i, x in enumerate(set(c2cl.clone))}
    tree.parent = tree.parent.map(cl2idx)
    tree.son = tree.son.map(cl2idx)
    print(tree)
    root = maketree_from_file(tree)
    showtree(root)
