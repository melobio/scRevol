import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import Manager
import argparse

def calculate_cell_accuracy(cell_to_cluster, tree_path, gt_pairs):
    acc_dif, acc_same = 0, 0
    total_num = 0
    for i in gt_pairs.keys():
        if i in cell_to_cluster.keys():
            c_i = cell_to_cluster[i]
            gt_children = gt_pairs[i]
            for child in gt_children:
                if child in cell_to_cluster.keys():
                    c_j = cell_to_cluster[child]
                    total_num += 1
                    if c_i in find_ancestors(tree_path, c_j):
                        acc_dif += 1
                    elif c_i == c_j:
                        acc_same += 1
    return acc_dif, acc_same, total_num

def find_ancestors(tree_path, node):
    ancestors = []
    current = node
    parent_dict = dict(zip(tree_path["son"], tree_path["parent"]))
    while current in parent_dict:
        parent = parent_dict[current]
        ancestors.append(parent)
        current = parent
    return ancestors

def load_gt_pairs_from_csv(filename='gt_pairs.csv'):
    df = pd.read_csv(filename)
    gt_pairs = {}
    for _, row in df.iterrows():
        children = eval(row['children'])
        gt_pairs[row['cell']] = children
    return gt_pairs

def process_datasets(root_dir, dataset_ids, method, data_type):
    results = []
    for dataset_id in dataset_ids:
        dataset_name = f'c{dataset_id}_{data_type}'
        data_path = f'{root_dir}/data/lineage_trace_data/m5k_lg{dataset_id}_character_matrix.alleleThresh.txt'
        gt_path = f'{root_dir}/data/lineage_trace_data/c{dataset_id}_CNV_gt_inherit.csv'
        res_dir = f'{root_dir}/rl_leiden/results/lineage_trace_data/{method}/{data_type}/leiden'

        # Load data
        data = pd.read_csv(data_path, sep="\t", index_col=0)
        data.index = data.index.str.strip()
        data = data.replace('-', np.nan)
        threshold = int(data.shape[1] * 0.7)
        data_cleaned = data.dropna(thresh=threshold)
        data_filled = data_cleaned.fillna(-1).astype(int)

        # Load ground truth pairs
        gt_pairs = load_gt_pairs_from_csv(filename=gt_path)

        # Load prediction results
        pred_labels = pd.read_csv(f'{res_dir}/{dataset_name}/cell2cluster.csv')
        tree_path = pd.read_csv(f'{res_dir}/{dataset_name}/tree_path.csv')
        cell_to_cluster = pred_labels.set_index('cell')['cluster']
        cell_to_cluster.index = cell_to_cluster.index.str.strip()

        # Compute accuracy
        acc_dif, acc_same, n_pairs = calculate_cell_accuracy(cell_to_cluster, tree_path, gt_pairs)
        accuracy = (acc_dif + acc_same) / n_pairs
        results.append({
            'dataset_id': dataset_id,
            'num_pairs': n_pairs,
            'same_correct': acc_same,
            'dif_correct': acc_dif,
            'accuracy': accuracy
        })

    # Compute mean and std
    accuracies = [res['accuracy'] for res in results]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    results.append({
        'dataset_id': 'mean_std',
        'num_pairs': '-',
        'same_correct': '-',
        'dif_correct': '-',
        'accuracy': f'{mean_accuracy:.4f} ± {std_accuracy:.4f}'
    })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    res_path = f'{res_dir}/all_tracing.csv'
    results_df.to_csv(res_path, index=False)
    print(f'Results saved to {res_path}')

def process_concat(root_dir, method, data_type):
    results = []
    res_dir = f'{root_dir}/rl_leiden/results/lineage_trace_data/{method}/{data_type}/leiden'
    # Load prediction results
    pred_labels_all = pd.read_csv(f'{res_dir}/concat_{data_type}/cell2cluster.csv')
    tree_path = pd.read_csv(f'{res_dir}/concat_{data_type}/tree_path.csv')
    meta = pd.read_csv('/home/ubuntu/duxinghao/clone/data/lineage_trace_data/concat_meta.csv', index_col=0)
    cell_to_cluster_all = pred_labels_all.set_index('cell')['cluster']
    cell_to_cluster_all.index = cell_to_cluster_all.index.str.strip()

    grouped_indices = meta.groupby('sample').apply(lambda x: x.index.tolist())
    for group, indices in grouped_indices.items():
        cell_to_cluster = cell_to_cluster_all[indices]
        # dataset_name = f'c{dataset_id}_{data_type}'
        dataset_name = group.replace('.csv', '')
        dataset_id = eval(dataset_name.split('_')[0].replace('c',''))
        # data_path = f'{root_dir}/data/lineage_trace_data/m5k_lg{dataset_id}_character_matrix.alleleThresh.txt'
        gt_path = f'{root_dir}/data/lineage_trace_data/c{dataset_id}_CNV_gt_inherit.csv'

        # Load ground truth pairs
        gt_pairs = load_gt_pairs_from_csv(filename=gt_path)

        # Compute accuracy
        acc_dif, acc_same, n_pairs = calculate_cell_accuracy(cell_to_cluster, tree_path, gt_pairs)
        accuracy = (acc_dif + acc_same) / n_pairs
        results.append({
            'dataset_id': dataset_id,
            'num_pairs': n_pairs,
            'same_correct': acc_same,
            'dif_correct': acc_dif,
            'accuracy': accuracy
        })

    # Compute mean and std
    accuracies = [res['accuracy'] for res in results]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    results.append({
        'dataset_id': 'mean_std',
        'num_pairs': '-',
        'same_correct': '-',
        'dif_correct': '-',
        'accuracy': f'{mean_accuracy:.4f} ± {std_accuracy:.4f}'
    })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    res_path = f'{res_dir}/concat_{data_type}/tracing.csv'
    results_df.to_csv(res_path, index=False)
    print(f'Results saved to {res_path}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_ids', type=str, default='17,22,26,29,30,39', help='Comma-separated list of dataset IDs')
    parser.add_argument('--method', type=str, default='rl_leiden', help='Clustering method')
    parser.add_argument('--data_type', type=str, default='CNV')
    parser.add_argument('--concat', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    root_dir = '/home/ubuntu/duxinghao/clone'
    dataset_ids = args.dataset_ids.split(',')
    method = args.method
    data_type= args.data_type

    if args.concat:
        process_concat(root_dir, method, data_type)
    else:
        process_datasets(root_dir, dataset_ids, method, data_type)
