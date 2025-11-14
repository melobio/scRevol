import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import precision_recall_curve, auc

from src.maketree import maketree
from src.maketree.from_file import maketree_from_file
from src.maketree.utils import get_ancestors, build_label_to_node_map, get_treenodes, showtree

def cal1B(truth, pred):
    return (truth + 1 - min(truth+1, abs(pred-truth))) / float(truth+1)

def cal2_modified(truth, pred):
    real_ccm = CCM(truth)
    pred_ccm = CCM(pred)
    iu = np.tril_indices_from(real_ccm, k=1)
    y_true = real_ccm[iu].ravel()
    y_score = pred_ccm[iu].ravel()

    aupr = compute_aupr(y_true, y_score)
    ajsd = np.array(jensenshannon(real_ccm, pred_ccm, axis=1) ** 2).mean()
    ajsd_normed = 1 - ajsd

    return (aupr + ajsd_normed) / 2

def cal3(truth, pred, cnv, tree_method='mst', real_tree=None, pred_tree=None):
    '''
        from nature: Pearson correlation coefficient between [CCM, ADM, ADM.T, CM]s
    '''
    # print("real tree:", real_tree)
    if real_tree is None:
        real_root = maketree(cnv, truth, method=tree_method)
    else:
        real_root = maketree_from_file(real_tree)
    # print("truth:", truth)
    # showtree(maketree(cnv,truth, method=tree_method))
    # showtree(maketree_from_file(real_tree))
    real_ccm = CCM(truth)
    real_adm = ADM(cnv, truth, real_root)
    real_cm  = CM(real_ccm, real_adm)
    real_matrix = np.concatenate([real_ccm, real_adm, real_adm.T, real_cm]).flatten()

    if pred_tree is None:   
        pred_root = maketree(cnv, pred, method=tree_method)
    else:
        pred_root = maketree_from_file(pred_tree)
    # print("pred")
    # showtree(maketree(cnv,pred, method=tree_method))
    # print("file")
    # showtree(maketree_from_file(pred_tree))
    pred_ccm = CCM(pred)
    pred_adm = ADM(cnv, pred, pred_root)
    pred_cm  = CM(pred_ccm, pred_adm)
    pred_matrix = np.concatenate([pred_ccm, pred_adm, pred_adm.T, pred_cm]).flatten()

    pcc = np.corrcoef(real_matrix, pred_matrix)[0, 1]
    return pcc

''' matrix calculation '''
def CCM(label):
    if isinstance(label, list):
        label = np.array(label)
    num_samples = label.shape[0]
    unique_clusters = np.unique(label)
    ccm_matrix = np.zeros((num_samples, num_samples), dtype=int)

    for cluster_id in unique_clusters:
        cluster_indices = np.where(label == cluster_id)[0]
        ccm_matrix[np.ix_(cluster_indices, cluster_indices)] = 1
    
    return ccm_matrix

def CM(ccm, adm):
    return 1 - ccm - adm - adm.T

def ADM(cnv, labels, root):
    m = len(cnv) 
    ADM = np.zeros((m, m), dtype=int)

    nodes = get_treenodes(root)
    label_to_node = build_label_to_node_map(nodes)

    cell_to_ancestors = {i:[] for i in range(m)}
    for i in range(m):
        cell_cluster_label = labels[i]
        cell_node = label_to_node[cell_cluster_label]
        ancestors = get_ancestors(cell_node)
        cell_to_ancestors[i].extend([a.label for a in ancestors])

    for i in range(m):
        for j in range(m):
            if labels[j] in cell_to_ancestors[i]:
                ADM[j, i] = 1

    return ADM

def compute_aupr(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    aupr = auc(recall, precision)
    
    return aupr
