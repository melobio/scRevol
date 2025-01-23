import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    dir1 = '/home/ubuntu/duxinghao/clone/rl_leiden/results/data_large2/leiden/CNV/leiden/'
    dir2 = '/home/ubuntu/duxinghao/clone/rl_leiden/results/data_large2/rl_leiden/CNV/leiden/'

    win, same, lose = 0, 0, 0
    for subdir in os.listdir(dir2):
        if not os.path.isdir(dir1+subdir):
            print(subdir)
            continue
        id = eval(subdir.split('_')[0])
        tar = pd.read_csv(dir1+subdir+'/result.csv', index_col=None)
        tar.columns = [col.replace(' ', '') for col in tar.columns] 
        res = pd.read_csv(dir2+subdir+'/result.csv', index_col=None)
        res.columns = [col.replace(' ', '') for col in res.columns] 
        cols    = ['nmi', 'ari', 'vm', 'sc1b', 'sc2','sc3']
        weights = [ 1,     1,     1,    1,      -1,    1] 
        res_result = np.round(res[cols].values*weights, 4)
        tar_result = np.round(tar[cols].values*weights, 4)
        if np.all(res_result == tar_result):
            same += 1
            print(f'Same {subdir}:', res[cols], tar[cols], sep='\n')
        elif np.all(res_result >= tar_result):
            win += 1
            print(f'Win {subdir}:', res[cols], tar[cols], sep='\n')
        else:
            lose += 1
            print(f'Lose {subdir}:', res[cols], tar[cols], sep='\n')
    
    print(f'Result: Win:{win}-Same:{same}-Lose:{lose}')
