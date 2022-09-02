import numpy as np
from sklearn.neighbors import KernelDensity

def fit_kde(x, bw=0.03):
    p = KernelDensity(bandwidth=bw, kernel='tophat')
    p.fit(x)
    return p

def MI(x, y):
    xy = np.concatenate([x, y], axis=-1)
    log_pxy = fit_kde(xy).score_samples(xy)
    log_px = fit_kde(x).score_samples(x)
    log_py = fit_kde(y).score_samples(y)
    log_ratio = log_pxy - log_px - log_py
    return np.mean(log_ratio)
