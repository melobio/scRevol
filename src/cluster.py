import os
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import anndata as ann
import scanpy as sc
from kneed import KneeLocator
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from torch.utils.tensorboard import SummaryWriter

from src.evaluation.evaluate import evaluate

class Normalizer(object):
    def __init__(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0) + 1e-5

    def normalize(self, data):
        data = (data - self.mean) / self.std
        return data

    def unnormalize(self, data):
        data = data * self.std + self.mean
        return data
    

class RLCluster(object):
    def __init__(self, 
                 algo,
                 tree_method,
                 data, 
                 c2cl, 
                 gt_tree,
                 cl2idx,
                 n_comps,
                 model,
                 optim,
                 log_dir,
                 device,
                 meta,
                 meta_col,
                 eval=True,
                 reward='sii',
        ):
        self.algo = algo
        self.adata = ann.AnnData(data.values, obs=np.arange(data.shape[0]), var=np.arange(data.shape[1]), dtype=float)
        if eval:
            self.c2cl = c2cl
            # cl2idx = {x:i for i, x in enumerate(set(self.c2cl.clone))}
            self.cl2idx = cl2idx
            self.true_labels = self.c2cl.clone.map(cl2idx).values
            self.true_tree = gt_tree
            # print(self.true_labels, self.true_tree, self.cl2idx)
        self.cnv_data = data.values
        os.makedirs(f'{log_dir}', exist_ok=True)

        self.log_dir = log_dir
        self.logger = SummaryWriter(f'{log_dir}')
        self.device = device

        sc.tl.pca(self.adata, n_comps=n_comps)
        self.pca_results = self.adata.obsm['X_pca']
        self.pca_results = torch.from_numpy(self.pca_results).float().to(device)
        self.pca_normalizer = Normalizer(self.pca_results)

        self.data = torch.from_numpy(data.values).float().to(device)
        self.data = (self.data - self.data.mean(0)) / (self.data.std(0) + 1e-5)
        self.data_numpy = self.data.detach().cpu().numpy()
        self.model = model
        self.optim = optim
        self.eval_mode = eval
        if meta is not None and meta_col is not None:
            meta_reset = meta.reset_index()
            self.grouped_indices = meta_reset.groupby(meta_col).apply(lambda x: x.index.tolist())
        else:
            self.grouped_indices = {None:np.arange(self.data.shape[0])}
        self.w1 = 1
        self.w2 = 0
        
        self.tree_method = tree_method

        if reward == 'sii':
            self.reward_func = silhouette_score
        elif reward == 'chi':
            self.reward_func = calinski_harabasz_score
        elif reward == 'dbi':
            self.reward_func = davies_bouldin_score

    def leiden(self, embed):
        if isinstance(embed, torch.Tensor):
            embed = embed.detach().cpu().numpy()
        adata = ann.AnnData(embed, obs=np.arange(embed.shape[0]), var=np.arange(embed.shape[1]))
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)
        pred_labels = adata.obs['leiden'].astype(int)
        return pred_labels.values
    
    def kmeans(self, embed):
        if isinstance(embed, torch.Tensor):
            embed = embed.detach().cpu().numpy()
        
        sse = []
        for num_clusters in list(range(1, 11)):
            kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=0)
            kmeans.fit_predict(embed)
            sse.append(kmeans.inertia_)
        kneedle = KneeLocator(np.arange(1, 11), sse, S=1, curve="convex", direction="increasing")
        n_c = kneedle.knee if kneedle.knee is not None else 9
        kmeans = KMeans(n_clusters=n_c, n_init='auto', random_state=0)
        kmeans.fit_predict(embed)
        
        return kmeans.labels_
    
    def get_reward(self, embedding, labels):
        ret = 0
        for group, indices in self.grouped_indices.items():
            if len(set(labels[indices])) > 1:
                ret += self.reward_func(self.data_numpy[indices], labels[indices])
        return ret 


    def get_dist(self):
        z = self.model(self.data)
        z_mean, z_logstd = torch.chunk(z, 2, -1)
        z_std = torch.clamp(z_logstd, min=-1, max=5).exp()
        z_dist = torch.distributions.Normal(z_mean, z_std)

        return z_dist
    
    def get_labels(self, embed):
        if self.algo == "leiden":
            labels = self.leiden(embed)
        elif self.algo == "kmeans":
            labels = self.kmeans(embed)
        else:
            raise NotImplementedError
        return labels

    def learn(self, bc_epochs, rl_epochs, samples_per_epoch, epsilon=0.2):
        self.best_embed = self.pca_results.detach().cpu().numpy()
        best_labels = self.get_labels(self.pca_results.detach().cpu().numpy())
        best_return = self.get_reward(self.pca_results.detach().cpu().numpy(), best_labels)
        self.logger.add_scalar('train/original_return', best_return, 0)
        normalized_pca = self.pca_normalizer.normalize(self.pca_results)

        if self.eval_mode:
            results = evaluate(
                cnv_data=self.cnv_data, 
                true_labels=self.true_labels, 
                pred_labels=best_labels, 
                tree_method=self.tree_method,
                real_tree=self.true_tree)
            df = pd.DataFrame.from_dict({k: [v] for k, v in results.items()})
            df.to_csv(os.path.join(self.log_dir, 'result.csv'))
        
        for e in tqdm(range(bc_epochs), desc='BC Training Epochs'):
            z_dist = self.get_dist()
            bc_loss = -z_dist.log_prob(normalized_pca).mean()

            self.optim.zero_grad()
            bc_loss.backward()
            self.optim.step()

            self.logger.add_scalar('train/init_bc_loss', bc_loss.item(), e)

        for e in tqdm(range(rl_epochs), desc='Training Epochs'):
            z_dist = self.get_dist()

            log_probs, rs, z_samples = [], [], []
            for _ in range(samples_per_epoch):
                with torch.no_grad():
                    z_sample = z_dist.sample()

                # get return
                embed = self.pca_normalizer.unnormalize(z_sample).detach().cpu().numpy()
                labels = self.get_labels(embed)
                r = self.get_reward(embed, labels)
                z_samples.append(z_sample.detach().cpu().numpy())
                log_probs.append(z_dist.log_prob(z_sample).mean().detach().cpu().numpy())
                rs.append(r)

            z_samples = torch.Tensor(np.array(z_samples)).to(self.device)
            log_probs = torch.Tensor(np.array(log_probs)).to(self.device)
            rs = torch.Tensor(np.array(rs)).to(self.device)
            advantage = rs - rs.mean()

            for _ in range(samples_per_epoch):
                z_dist = self.get_dist()
                ratio = z_dist.log_prob(z_samples).mean(dim=(1, 2)) / log_probs
                rl_loss = torch.min(ratio * advantage, ratio.clip(1-epsilon, 1+epsilon) * advantage).mean()
                bc_loss = -z_dist.log_prob(normalized_pca).mean()
                loss = rl_loss + 1e-3 * ratio.mean() * bc_loss 
                # loss = rl_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            embed = self.pca_normalizer.unnormalize(self.get_dist().mean).detach().cpu().numpy()
            labels = self.get_labels(embed)
            r = self.get_reward(embed, labels)

            if self.eval_mode:
                # results = self.evaluate(labels)
                results = evaluate(
                    cnv_data=self.cnv_data, 
                    true_labels=self.true_labels, 
                    pred_labels=labels, 
                    tree_method=self.tree_method,
                    real_tree=self.true_tree)
                for k, v in results.items():
                    self.logger.add_scalar(f'eval/{k}', v, e)

            if r > best_return:
                best_return = r
                best_labels = labels
                self.best_embed = embed
                if self.eval_mode:
                    df = pd.DataFrame.from_dict({k: [v] for k, v in results.items()})
                    df.to_csv(os.path.join(self.log_dir, 'result.csv'))

            self.logger.add_scalar('train/rl_loss', rl_loss.item(), e)
            self.logger.add_scalar('train/bc_loss', rl_loss.item(), e)
            self.logger.add_scalar('train/best_return', best_return, e)
            self.logger.add_scalar('train/return', r, e)

        return best_labels



