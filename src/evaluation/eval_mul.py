import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import v_measure_score as vm_score

if __name__ == "__main__":
    method = 'leiden'
    cols = ["celltype", "Stage", "sample", "anatomical_location", "anatomical_location_modified"]  # 新增一列
    scorings = { 
        'ari': ari_score,
        'nmi': nmi_score,
        'vm': vm_score,
               }
    for sc in scorings.keys():
        res = {}
        scoring = scorings[sc]
        for i in [1, 2, 5, 6, 7]:
            idx = f'data_{i}'
            res[idx] = []
            meta = pd.read_csv(f'/path/to/your/data/data{i}_meta.csv', index_col=0)
            pred_label = pd.read_csv(f'./clone/rl_leiden/results/CNV_multiSample_martix/{method}/CNV/leiden/data{i}_CNV/cell2cluster.csv', index_col=0).values.reshape(-1)
            
            for col in ["celltype", "Stage", "sample", "anatomical_location"]:
                true_label = meta[col].values.reshape(-1)
                score = scoring(true_label, pred_label)
                res[idx].append(score)
            
            # 修改 'anatomical_location' 中非 'ovary' 的值
            true_label = meta['anatomical_location'].values.reshape(-1)
            true_label[true_label != 'Ovary'] = 'other'
            ari_modified = scoring(true_label, pred_label)
            res[idx].append(ari_modified)  # 将新的ARI结果添加到列表中
        
        res = pd.DataFrame(res).T
        res.columns = cols
        res.to_csv(f'./rl_leiden/results/CNV_multiSample_martix/{method}/CNV/leiden/column_{sc}.csv')
