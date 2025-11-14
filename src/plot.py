import networkx as nx
from matplotlib import pyplot as plt

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