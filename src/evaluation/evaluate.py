import argparse
import numpy as np

from src.metrics import cal1B, cal2_modified, cal3
from src.utils import load_data
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import v_measure_score as vm_score


def evaluate(cnv_data, true_labels, pred_labels, tree_method='mst', real_tree=None, pred_tree=None):
    """
    Evaluate clustering performance using various metrics.

    Parameters:
    - cnv_data: numpy array, copy number variation data. (N * D)
    - true_labels: array-like, true cluster labels. (N, )
    - pred_labels: array-like, predicted cluster labels. (N, )

    Returns:
    - dict: containing scores for NMI, ARI, VM, and custom metrics lcc, pcd, tc.
    """
    nmi = nmi_score(true_labels, pred_labels, average_method='arithmetic')
    ari = ari_score(true_labels, pred_labels)
    vm = vm_score(true_labels, pred_labels)
    lcc = cal1B(truth=len(set(true_labels)), pred=len(set(pred_labels)))
    # sc2 = cal2(truth=true_labels, pred=pred_labels)
    pcd = cal2_modified(truth=true_labels, pred=pred_labels)
    tc = cal3(truth=true_labels, pred=pred_labels, cnv=cnv_data, tree_method=tree_method, real_tree=real_tree, pred_tree=pred_tree)
    return {'nmi': nmi, 'ari': ari, 'vm': vm, 'lcc': lcc,'pcd': pcd, 'tc': tc}


def get_args():
    """
    Parse command line arguments.

    Returns:
    - argparse.Namespace: containing data directory and dataset ID.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/data_large2/')
    parser.add_argument('--dataset_id', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = get_args()
    
    # Load data and true labels
    data, c2cl, dataset_name = load_data(args.data_dir, args.dataset_id)
    
    # Map true labels to indices
    cl2idx = {x:i for i, x in enumerate(set(c2cl.clone))}
    true_labels = c2cl.clone.map(cl2idx).values
    
    # Extract CNV data
    cnv_data = data.values

    # Placeholder for predicted labels (currently all zeros)
    pred_labels = np.zeros_like(true_labels)
    
    # Evaluate the clustering performance
    results = evaluate(cnv_data, true_labels, pred_labels)
    
    # Print the evaluation results
    print(results)