import pandas as pd

if __name__ == '__main__':
    # 定义需要删除的簇
    nonepi_cls = [4,5,2,0,8,9]
    
    name = 'SOL1307'
    # 加载数据
    c2cl = pd.read_csv(f'/home/ubuntu/duxinghao/clone/rl_leiden/results/SingleSample_CNV/rl_leiden/CNV/leiden/{name}_CNV/cell2cluster.csv', index_col=0)
    cnv = pd.read_csv(f'/home/ubuntu/duxinghao/clone/data/SingleSample_CNV/{name}_CNV.csv', index_col=0).T
    meta = pd.read_csv(f'/home/ubuntu/duxinghao/clone/data/SingleSample_CNV/{name}_meta.csv', index_col=0)
    # print(cnv.shape, meta.shape),quit()
    print(f"Before filter CNV shape: {cnv.shape}")

    # 找到需要删除的细胞名
    cells_to_remove = c2cl[c2cl['cluster'].isin(nonepi_cls)].index

    # 从 cnv 和 meta 数据中删除这些细胞行
    cnv_filtered = cnv.drop(index=cells_to_remove, errors='ignore')
    meta_filtered = meta.drop(index=cells_to_remove, errors='ignore')

    # 输出结果（或保存）
    print(f"Filtered CNV shape: {cnv_filtered.shape}")
    print(f"Filtered Meta shape: {meta_filtered.shape}")
    
    # 保存到新的文件（如果需要）
    cnv_filtered.T.to_csv(f'/home/ubuntu/duxinghao/clone/data/SingleSample_CNV/{name}filtered_CNV.csv')
    meta_filtered.to_csv(f'/home/ubuntu/duxinghao/clone/data/SingleSample_CNV/{name}filtered_meta.csv')
