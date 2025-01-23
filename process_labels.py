import numpy as np
import pandas as pd
import argparse

from utils import get_root_data, maketree, l2_distance, get_parent_child_pairs, drawtree, load_real_data

def replace_outliers_with_median(df):
    df_new = pd.DataFrame(data=df.values, columns=df.columns)
    for col in df.columns:
        Q1 = df[col].quantile(0.25)  # 计算Q1
        Q3 = df[col].quantile(0.75)  # 计算Q3
        IQR = Q3 - Q1  # 计算四分位距
        
        # 定义异常值的范围
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 计算中位数
        median = df[col].median()
        
        # 将超出范围的值替换为中位数
        df_new[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), median, df[col])
    return df_new

def relabel_nonepi(cnv, label):
    new_label = label.values.copy()
    cnv_produced = replace_outliers_with_median(cnv)
    root_val, _ = get_root_data(cnv_produced)
    distances = np.sum(cnv_produced != root_val, axis=1)
    new_label[distances < 1.4 * distances.mean()] = -2
    return pd.DataFrame(data=new_label, index=label.index, columns=label.columns)

def remove_nonepi(cnv, label, meta):
    new_label = label.values.copy()
    new_label[meta['celltype']=='non-Epi'] = -2
    return pd.DataFrame(data=new_label, index=label.index, columns=label.columns)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='CNV_multiSample_martix')
    parser.add_argument('--dataname', type=str, default='SOL003')
    parser.add_argument('--method', type=str, default='leiden')
    parser.add_argument('--data_type', type=str, default='CNV')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    root_dir = '/home/ubuntu/duxinghao/clone'
    dataset_id = args.dataname
    task = args.task_type
    method = args.method

    # Example usage:
    cnv, _, _, meta = load_real_data(f'{root_dir}/data/{task}', f'{dataset_id}_CNV')
    c2cl = pd.read_csv(f'{root_dir}/rl_leiden/results/{task}/{method}/CNV/leiden/{dataset_id}_CNV/cell2cluster.csv', index_col=0)
    # tree_path = pd.read_csv(f'{root_dir}/rl_leiden/results/{task}/rl_leiden/CNV/leiden/c{dataset_id}_CNV/tree_path.csv', index_col=0)

    # new_label = relabel_nonepi(cnv, c2cl)
    new_label = remove_nonepi(cnv, c2cl, meta)
    new_label.to_csv(f'{root_dir}/rl_leiden/results/{task}/{method}/CNV/leiden/{dataset_id}_CNV/c2cl_new.csv')

    root = maketree(cnv=cnv[new_label.values!=-2].values, labels=new_label[new_label.values!=-2].values, dist_func=l2_distance)
    # root = maketree(cnv=cnv.values, labels=c2cl.values, dist_func=l2_distance)
    # showtree(root)
    tree_df = pd.DataFrame(data=get_parent_child_pairs(root), columns=['parent', 'son'])
    tree_df.to_csv(f'{root_dir}/rl_leiden/results/{task}/{method}/CNV/leiden/{dataset_id}_CNV/tree_path_new.csv', index=None)
    drawtree(tree_df, f'{root_dir}/rl_leiden/results/{task}/{method}/CNV/leiden/{dataset_id}_CNV/tree_new.pdf')