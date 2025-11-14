import numpy as np
from scipy.stats import mode

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

def merge_nodes(nodes):
    assert sum([0 if node.parent is None else 1 for node in nodes]) == 0 # 只有根能合并
    new_node = Node(v=0, n=0, label=nodes[0].label)
    for node in nodes:
        new_node.n += node.n
        for offspring in node.offsprings:
            new_node.add_offspring(offspring)
            offspring.set_parent(new_node)
    return new_node

def get_center(cnv):
    # return np.mean(cnv, axis=0) # use mean as center
    most_vec = mode(cnv, axis=0, keepdims=True).mode.flatten()
    return np.array(most_vec)
    