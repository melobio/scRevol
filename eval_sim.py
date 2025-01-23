import os
import pandas as pd

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_summary_statistics(results_folder, num_files=500):
    # Initialize an empty list to hold the data from each file
    all_metrics = []

    # Loop through all the files
    for dir in os.listdir(results_folder):
        if not os.path.isdir(os.path.join(results_folder, dir)):
            continue
        file_path = os.path.join(results_folder, f"{dir}/result.csv")
        f_id = eval(dir.split('_')[0])
        
        if os.path.exists(file_path):
            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
                # The file contains only one row, so we can extract it directly
                metrics = df.iloc[0][['nmi', 'ari', 'vm', 'sc1b', 'sc2', 'sc3']].to_dict()
                metrics['file_id'] = f_id  # Adding the file_id to track the source
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Convert the collected data into a DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Calculate mean and variance for each metric
    summary_df = metrics_df[['nmi', 'ari', 'vm', 'sc1b', 'sc2', 'sc3']].agg(['mean', 'var'])

    return summary_df

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='leiden')
    parser.add_argument('--data_type', type=str, default='CNV')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    # Path to the results folder
    results_folder = f'/home/ubuntu/duxinghao/clone/rl_leiden/results/data_large2/{args.method}/{args.data_type}/leiden/'

    # Calculate the summary statistics (mean and variance)
    summary_df = calculate_summary_statistics(results_folder)

    # Save the summary statistics to a CSV file
    output_csv_path = os.path.join(results_folder, 'summary_statistics.csv')
    ensure_directory_exists(os.path.dirname(output_csv_path))
    summary_df.to_csv(output_csv_path)
    print(f"Summary statistics saved to {output_csv_path}")
