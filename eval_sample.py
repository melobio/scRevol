import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='CNV_multiSample_martix')
    parser.add_argument('--data_names', type=str, default='data1,data2,data3,data5,data6,data7', help='Comma-separated list of dataset names')
    parser.add_argument('--method', type=str, default='leiden')
    parser.add_argument('--data_type', type=str, default='CNV')
    parser.add_argument('--meta_columns', type=str, default='celltype,Stage,sample,anatomical_location', help='Comma-separated list of meta columns to process')
    return parser.parse_args()

def process_dataset(task_type, data_name, method, data_type, meta_columns):
    # Load data
    type_prefix = data_type.split('_')[0]
    meta_df_path = f'/home/ubuntu/duxinghao/clone/data/{task_type}/{data_name}_meta.csv'
    cell2cluster_path = f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{task_type}/{method}/{data_type}/leiden/{data_name}_{type_prefix}/cell2cluster.csv'

    try:
        meta_df = pd.read_csv(meta_df_path, index_col=0)
        meta_df.columns = [col.strip() for col in meta_df]
        meta_df.index = meta_df.index.str.strip()
        cell2cluster_df = pd.read_csv(cell2cluster_path, index_col=0)
    except FileNotFoundError as e:
        print(f"Error loading data for {data_name}: {e}")
        return [{'data_name': data_name, 'meta_column': 'N/A', 'average_purity': 'N/A', 'plot_path': f'Error: {e}'}]

    # Merge data to associate cells with clusters
    meta_df.reset_index(inplace=True)
    meta_df.rename(columns={'index': 'cell'}, inplace=True)
    merged_df = meta_df.merge(cell2cluster_df, on='cell', how='inner')

    results = []

    for meta_column in meta_columns:
        # Compute counts of each category per cluster node
        node_category_counts = merged_df.groupby(['cluster', meta_column]).size().unstack(fill_value=0)

        # Calculate proportions for each category in each cluster
        node_ratios = node_category_counts.div(node_category_counts.sum(axis=1), axis=0)

        # Calculate purity for each node (maximum category proportion)
        node_purity = node_ratios.max(axis=1)

        # Calculate weighted average purity across all nodes
        total_cells_per_node = node_category_counts.sum(axis=1)
        average_purity = (node_purity * total_cells_per_node).sum() / total_cells_per_node.sum()

        # Visualize proportions
        output_dir = f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{task_type}/{method}/{data_type}/leiden/{data_name}_{type_prefix}'
        ensure_directory_exists(output_dir)

        node_ratios_sorted = node_ratios.sort_values(by=node_ratios.columns[0], ascending=False)
        ax = node_ratios_sorted.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')

        plt.title(f'{meta_column} Proportion in Each Cluster Node\n(Average Purity: {average_purity:.4f})', fontsize=14)
        plt.xlabel('Cluster Node')
        plt.ylabel('Proportion')
        plt.xticks(rotation=0)
        plt.legend(title=meta_column, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save the plot
        plot_path = os.path.join(output_dir, f'{meta_column}_proportions.pdf')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        results.append({
            'data_name': data_name,
            'meta_column': meta_column,
            'average_purity': average_purity,
            'plot_path': plot_path
        })

    return results

def calculate_summary_statistics(all_results):
    # Create a DataFrame from all results
    results_df = pd.DataFrame(all_results)

    # Remove rows where 'average_purity' is 'N/A'
    results_df = results_df[results_df['average_purity'] != 'N/A']

    # Convert 'average_purity' to numeric values, forcing errors to NaN
    results_df['average_purity'] = pd.to_numeric(results_df['average_purity'], errors='coerce')

    # Group by meta_column and calculate mean and variance of average_purity
    summary_df = results_df.groupby('meta_column')['average_purity'].agg(['mean', 'var']).reset_index()

    return summary_df

if __name__ == '__main__':
    args = get_args()
    task_type = args.task_type
    data_names = args.data_names.split(',')
    method = args.method
    data_type = args.data_type
    meta_columns = args.meta_columns.split(',')

    all_results = []
    for data_name in data_names:
        dataset_results = process_dataset(task_type, data_name, method, data_type, meta_columns)
        if dataset_results is not None:
            all_results.extend(dataset_results)

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    output_csv_path = f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{task_type}/{method}/{data_type}/leiden/summary_results.csv'
    ensure_directory_exists(os.path.dirname(output_csv_path))
    results_df.to_csv(output_csv_path, index=False)
    print(f'Results saved to {output_csv_path}')

    # Calculate summary statistics (mean and variance of purity for each meta column)
    summary_df = calculate_summary_statistics(all_results)

    # Save the summary statistics to a new CSV
    summary_csv_path = f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{task_type}/{method}/{data_type}/leiden/summary_statistics.csv'
    ensure_directory_exists(os.path.dirname(summary_csv_path))
    summary_df.to_csv(summary_csv_path, index=False)
    print(f'Summary statistics saved to {summary_csv_path}')
