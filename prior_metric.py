import pandas as pd
import numpy as np
import json
import os
import argparse

from joblib import Parallel, delayed
from collections import defaultdict

from net import *
from utils import *
from cluster import *

def save_dict(dic, savepath):
    dic_str = {str(k):dic[k] for k in dic.keys()}
    dic_json = json.dumps(dic_str, sort_keys=False, indent=4, separators=(',', ':'))
    with open(f'{savepath}.json', 'w') as f:
        f.write(dic_json)


def replace_outliers_with_median(df):
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
        df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), median, df[col])

def calculate_intersection_difference(parent_values, child_values, genes):
    # Use numpy logical operations to avoid Python loops
    intersection_mask = (parent_values == child_values) & (parent_values != 2)
    intersection_count = np.sum(intersection_mask)
    
    parent_non_2_genes_count = np.sum(parent_values != 2)
    intersection_ratio = intersection_count / parent_non_2_genes_count if parent_non_2_genes_count > 0 else 0
    
    # Identify the difference genes efficiently
    difference_mask = (child_values != parent_values) & (child_values != 2)
    difference_genes = genes[difference_mask]
    
    return intersection_ratio, difference_genes

def calculate_differences(child1_values, child2_values, genes):
    # 左差集 (ss1 - ss2) 和 右差集 (ss2 - ss1)
    left_diff_genes = genes[(child1_values != child2_values) & (child1_values != 2)]
    right_diff_genes = genes[(child2_values != child1_values) & (child2_values != 2)]
    return set(left_diff_genes), set(right_diff_genes)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='SingleSample_CNV')
    parser.add_argument('--data_name', type=str, default='SOL1307')
    parser.add_argument('--method', type=str, default='leiden')
    parser.add_argument('--data_type', type=str, default='CNV')
    parser.add_argument('--meta_column', type=str, default='celltype', help='Column in meta_df to calculate proportions')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ## 读取用于预测模型的CNV矩阵，行为细胞，列为特征
    args = get_args()
    for f in os.listdir(f'data/data_large_exp'):
        f_id = eval(f.split('_')[0])
        if f_id == args.dataset_id:
            dataset_name = f
            break
    else:
        quit()
    type_prefix = args.data_type.split('_')[0]
    clean_mtx, _, dataset_name, _ = load_real_data(args.data_dir, args.data_name, remove_no_epi=args.remove_non_epi)
    c2c = pd.read_csv(f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{args.task_type}/{args.method}/{args.data_type}/leiden/{args.data_name}_{type_prefix}/cell2cluster.csv')
    c2c.columns = ['parent', 'son']

    tree = pd.read_csv(f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{args.task_type}/{args.method}/{args.data_type}/leiden/{args.data_name}_{args.data_type}/tree_path.csv')
    tree.columns = ['parent', 'son']

    path = defaultdict(list)      ## 簇树
    # p2s = defaultdict(list)       ## cluster - cell
    p2s = c2c.groupby('parent')['son'].apply(list).to_dict()

    clusters = set()              ## cluster
    for _, r in tree.iterrows():
        path[r['parent']].append(r['son'])
        clusters.add(r['parent'])
        clusters.add(r['son'])
        
    modified_mtx = dict()

    for x in clusters:
        tmp_mtx = clean_mtx.loc[p2s[x], :]
        replace_outliers_with_median(tmp_mtx)
        modified_mtx[x] = tmp_mtx

    print(path)
    for p in path:
        for s in path[p]:
            print(p, modified_mtx[p].shape,  s, modified_mtx[s].shape)
        
    # Prepare results dictionary
    results = {}

    # Iterate through paths and calculate using optimized parallelism
    for p, children in path.items():
        parent_df = modified_mtx[p]  # Parent node DataFrame
        
        for s in children:
            results[s] = {'p1': 0, 'p2': 0}
            child_df = modified_mtx[s]  # Child node DataFrame
            
            # Vectorized calculation of intersection and difference ratios using parallel processing
            parallel_results = Parallel(n_jobs=-1, backend='loky')(
                delayed(calculate_intersection_difference)(
                    parent_df.loc[parent_cell].values, 
                    child_df.loc[child_cell].values, 
                    parent_df.columns
                ) for parent_cell in parent_df.index for child_cell in child_df.index
            )
            
            # Aggregate results efficiently
            all_differences_union = set()
            intersection_ratios = []
            difference_ratios = []
            
            # Unpack parallel results directly using list comprehension
            intersection_ratios = [res[0] for res in parallel_results]
            all_differences_union = set.union(*(set(res[1]) for res in parallel_results))
            
            # Calculate difference ratios for each result
            union_size = len(all_differences_union)
            if union_size > 0:
                difference_counts = [len(res[1]) for res in parallel_results]
                difference_ratios = [x / union_size for x in difference_counts]
            
            # Store average results in the dictionary
            results[s]['p1'] = np.mean(intersection_ratios) if intersection_ratios else 0
            results[s]['p2'] = np.mean(difference_ratios) if difference_ratios else 0
            
            # Special handling for the 'root' node
            if p == 'root':
                results[s]['p1'] = None
            

    for p in path:
    # 子节点之间的分析
        ss = path[p]  # 所有子节点
        child_analysis = {}
        
        for i, ss1 in enumerate(ss):
            if 'c' not in results[ss1]:
                results[ss1]['c'] = []
            for j, ss2 in enumerate(ss):
                if i >= j:
                    continue  # Skip duplicate pairs or self-comparisons

                if 'c' not in results[ss2]:
                    results[ss2]['c'] = []

                child1_df = modified_mtx[ss1]
                child2_df = modified_mtx[ss2]
                
                # 并行处理子节点之间的细胞差集
                parallel_results = Parallel(n_jobs=-1)(delayed(calculate_differences)(
                    child1_df.loc[cell1].values, child2_df.loc[cell2].values, child1_df.columns
                ) for cell1 in child1_df.index for cell2 in child2_df.index)
                
                
                left_differences_union = set()
                right_differences_union = set()
                left_differences_results = list()
                right_differences_results = list()
                
                for left_diff_genes, right_diff_genes in parallel_results:
                    left_differences_results.append(len(left_diff_genes))
                    right_differences_results.append(len(right_diff_genes))
                    left_differences_union.update(left_diff_genes)
                    right_differences_union.update(right_diff_genes)

                left_union_size = len(left_differences_union)
                right_union_size = len(right_differences_union)

                left_rates = []
                right_rates = []

                if left_union_size > 0:
                    for x in left_differences_results:
                        left_rates.append(x / left_union_size)

                if right_union_size > 0:
                    for x in right_differences_results:
                        right_rates.append(x / right_union_size)

                # 计算子节点之间的左、右差集比例的均值
                average_left_rate = np.mean(left_rates) if left_rates else 0
                average_right_rate = np.mean(right_rates) if right_rates else 0
        
                results[ss1]['c'].append(average_left_rate)
                results[ss2]['c'].append(average_right_rate)

    save_dict(results, savepath=f'results/data_large_exp/aekmeans/{dataset_name}/prior_metric')
    plt.savefig(f'/home/ubuntu/duxinghao/clone/rl_leiden/results/{args.task_type}/leiden/{args.data_type}/leiden/{args.data_name}_{args.data_type}/prior_metric')