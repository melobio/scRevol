import numpy as np

def l1_distance(a, b):
    return np.abs(a-b).sum()

def l2_distance(a, b):
    return np.square(a - b).mean()
