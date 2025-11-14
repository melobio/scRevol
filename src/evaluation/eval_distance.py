import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def calculate_cluster_distances(data_file, cluster_file, tree_path, emb_file, use_emb=False):
    data = pd.read_csv(data_file, index_col=0).T
    data.index = data.index.str.strip()
    cluster_df = pd.read_csv(cluster_file, index_col=0)  
    tree_df = pd.read_csv(tree_path, index_col=0)

    if use_emb:
        emb_df  = pd.read_csv(emb_file, index_col=0)
        data = emb_df
    root_label = tree_df.index[0]

    root_v = data[cluster_df['cluster'] == root_label].values.mean(axis=0)

    distances = {}
    labels = np.unique(tree_df.values)
    for label in labels:
        parent = tree_df[tree_df['son'] == label].index[0]
        # parent_v = data[cluster_df['cluster'] == parent].values.mean(axis=0)
        node_df = data[cluster_df['cluster'] == label]
        if not node_df.empty:
            node_v = node_df.values.mean(axis=0)
            dist = np.sqrt(np.mean(np.square(root_v - node_v)))
            distances[label] = dist
        else:
            distances[label] = np.nan  

    return distances

def drawtree(tree_path, distances, path=None):
    df = pd.read_csv(tree_path)
    G = nx.DiGraph()
    G.add_edges_from(zip(df['parent'], df['son']))

    nodes = list(G.nodes)
    dist_values = [distances.get(node, 0) for node in nodes] 
    cmap = cm.inferno_r 
    norm = plt.Normalize(vmin=min(dist_values), vmax=max(dist_values))
    node_colors = [cmap(norm(dist)) if not np.isnan(dist) else "lightgrey" for dist in dist_values]

    plt.figure(figsize=(10, 8))
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot') 
    nx.draw(
        G, pos, with_labels=True, arrows=False, node_size=1500,
        node_color=node_colors, font_size=10, font_weight="bold", cmap=cmap
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Distance to Root', fontsize=12)

    plt.title("Tree Structure with Distances", fontsize=14)
    if path:
        plt.savefig(path)
        print(f'figure saved at {path}')

if __name__ == '__main__':
    root_dir = '.'
    dataset_id = '17'
    method = 'rl_leiden'

    # Example usage:
    data_file = f'{root_dir}/data/lineage_trace_data/c{dataset_id}_CNV.csv'
    cluster_file = f'{root_dir}/rl_leiden/results/lineage_trace_data/{method}/CNV/leiden/c{dataset_id}_CNV/cell2cluster.csv'
    tree_path = f'{root_dir}/rl_leiden/results/lineage_trace_data/{method}/CNV/leiden/c{dataset_id}_CNV/tree_path.csv'
    emb_file = f'{root_dir}/rl_leiden/results/lineage_trace_data/{method}/CNV/leiden/c{dataset_id}_CNV/embeddings.csv'

    # Calculate distances
    distances = calculate_cluster_distances(data_file, cluster_file, tree_path, emb_file, use_emb=False)

    # Draw tree with distances
    drawtree(tree_path, distances, path=f'{root_dir}/rl_leiden/results/lineage_trace_data/{method}/CNV/leiden/c{dataset_id}_CNV/tree_with_dist.pdf')
