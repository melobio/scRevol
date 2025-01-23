import os
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/home/ubuntu/duxinghao/clone/data/lineage_trace_data')
    parser.add_argument('--data_type', type=str, default='CNV')
    return parser.parse_args()

def concatenate_cnv_files(input_dir, output_file, args):
    cnv_files = [f for f in os.listdir(input_dir) if f.endswith(f'{args.data_type}.csv')]
    if not cnv_files:
        raise ValueError(f"No {args.data_type} CSV files found in the specified directory.")
    cnv_files = sorted(cnv_files)
    concatenated_data = []
    meta_data = []

    for file in cnv_files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path, index_col=0)
        print(file_path, df.shape)
        df.index = df.index.str.strip()

        # Append the origin file name to each column name for tracking
        df.columns = [col.strip() for col in df.columns]
        meta_data.extend([(col, file) for col in df.columns])

        # Concatenate data
        concatenated_data.append(df)
        
    # Create the meta DataFrame
    concatenated_data = pd.concat(concatenated_data, axis=1, join='outer')

    # Fill missing values with global mode
    global_mode = concatenated_data.mode().iloc[0]
    concatenated_data.fillna(global_mode, inplace=True)

    meta_df = pd.DataFrame(meta_data, columns=["", "sample"])
    print(meta_df.shape, concatenated_data.shape)
    
    # Save concatenated data and meta information
    concatenated_data.to_csv(f'{output_file}/concat_{args.data_type}.csv')
    meta_df.to_csv(f"{output_file}/concat_meta.csv", index=0)
    print(f"Files concatenated and saved at .csv and {output_file}/concat_meta.csv")

if __name__ == '__main__':
    args = get_args()
    concatenate_cnv_files(args.input_dir, args.input_dir, args)
