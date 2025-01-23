import torch
import argparse
from tqdm import tqdm

from net import *
from utils import *
from cluster import *

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='leiden')
    parser.add_argument('--data_dir', type=str, default='../data/data_large2/')
    parser.add_argument('--dataset_id', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results_leiden/')
    parser.add_argument('--n_components', type=int, default=10, help='pca components.')
    parser.add_argument('--hidden_dims', type=int, default=[512, 512], help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--rl_epochs', type=int, default=10, help='Number of epochs to train E.')
    parser.add_argument('--bc_epochs', type=int, default=100, help='Number of epochs to train bc.')
    parser.add_argument('--samples_per_epoch', type=int, default=100, help='Replay buffer size')
    parser.add_argument('--epsilon', type=float, default=0.2, help='rl clip threshold')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--eval_mode', action='store_false', default=True)
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--remove_non_epi', action='store_true', default=False)
    parser.add_argument('--meta_col', type=str, default=None)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device)

    # init data & logger
    set_seed_everywhere(args.seed)
    meta = None
    if args.eval_mode:
        data, c2cl, dataset_name = load_data(args.data_dir, args.dataset_id)
        data = [data]
    else:
        data, c2cl, dataset_name, meta = load_real_data(args.data_dir, args.data_name, remove_no_epi=args.remove_non_epi)
    log_dir = os.path.join(args.output_dir, args.algo, dataset_name)
    
    # init model & optimizer
    data_size = data.values.shape[0]
    input_dim = data.values.shape[1]
    model = MLP(input_dim, args.hidden_dims, args.n_components*2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # get algorithm
    rl_cluster = RLCluster(args.algo, data, c2cl, args.n_components, model, optimizer, log_dir, device, eval=args.eval_mode, meta=meta, meta_col=args.meta_col)

    # learn
    pred_labels = rl_cluster.learn(args.bc_epochs, args.rl_epochs, args.samples_per_epoch, args.epsilon)

    embed_df = pd.DataFrame(rl_cluster.best_embed, index=data.index)
    embed_df.to_csv(os.path.join(log_dir, 'embeddings.csv'))

    label_df = pd.DataFrame({'cell':data.index, 'cluster':pred_labels})
    label_df.to_csv(os.path.join(log_dir, 'cell2cluster.csv'), index=None)
    # make tree
    root = maketree(cnv=data.values, labels=pred_labels, dist_func=l2_distance)
    # showtree(root)
    tree_df = pd.DataFrame(data=get_parent_child_pairs(root), columns=['parent', 'son'])
    tree_df.to_csv(os.path.join(log_dir, 'tree_path.csv'), index=None)
    drawtree(tree_df, os.path.join(log_dir, 'tree.pdf'))
