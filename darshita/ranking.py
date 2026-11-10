"""
ranking.py - the three ranking-based band selection methods, as plain functions.

Every function takes `pixels`: a table with one row per pixel and one column per band
(shape: n_pixels x n_bands), and returns one score per band. Higher score = more useful band.
Steps 2 and 3 import these functions.
"""
import numpy as np


# ================================================================ method 1
def variance_scores(pixels):
    """Variance of each band. Divide by one global maximum first, so every band is on the
    same scale but bright bands stay brighter (that difference is what variance measures)."""
    scaled = pixels / pixels.max()
    return scaled.var(axis=0)              # axis=0 -> one value per column (band)


# ================================================================ method 2
def entropy_scores(pixels, n_bins=256):
    """Shannon entropy of each band:  H = - sum( p * log2(p) )
    where p = fraction of pixels falling in each histogram bin."""
    scores = np.zeros(pixels.shape[1])
    for b in range(pixels.shape[1]):
        counts, _ = np.histogram(pixels[:, b], bins=n_bins)
        p = counts / counts.sum()          # counts -> probabilities
        p = p[p > 0]                       # log(0) is undefined, empty bins add nothing
        scores[b] = -(p * np.log2(p)).sum()
    return scores


# ================================================================ method 3
def mvpca_scores(pixels, k):
    """MVPCA: score_i = sum over the first k components of  eigenvalue * (loading of band i)^2
    Bands are standardised first (mean 0, std 1) so no band wins just by being bright."""
    standardised = (pixels - pixels.mean(axis=0)) / pixels.std(axis=0)
    covariance = np.cov(standardised, rowvar=False)          # 200 x 200 band-to-band matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)   # eigh: for symmetric matrices
    order = np.argsort(eigenvalues)[::-1]                    # largest eigenvalue first
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    loadings = eigenvectors[:, :k] ** 2                      # how much each band feeds each component
    return loadings @ eigenvalues[:k]


# ============================================================ select top K
def top_k(scores, k):
    """Indices of the k largest scores, returned in band order."""
    best_first = np.argsort(scores)[::-1]                    # sort high -> low
    return sorted(best_first[:k].tolist())
