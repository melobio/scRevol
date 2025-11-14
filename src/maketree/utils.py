import numpy as np

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
    

def get_treenodes(root):
    nodes = []
    nodes_haschild = [root]
    while len(nodes_haschild) > 0:
        node = nodes_haschild.pop()
        nodes_haschild.extend(node.offsprings)
        nodes.append(node)
    return nodes