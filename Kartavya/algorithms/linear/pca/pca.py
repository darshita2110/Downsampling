"""
Principal Component Analysis written from scratch with NumPy.

Idea in one paragraph
---------------------
Every pixel is a point in C-dimensional space (C = number of channels). When channels are
correlated, the points lie close to a lower-dimensional flat surface. PCA finds new axes
(principal components) that are
  1. orthogonal (at right angles) to each other, and
  2. ordered by how much of the data's variance lies along them.
Keeping only the first K axes and dropping the rest loses the least possible variance - equivalently
it gives the smallest mean squared reconstruction error of any K-dimensional linear projection.

Steps (X is N pixels x C channels)
----------------------------------
  mu  = mean of every column                      -> centre the data:  Xc = X - mu
  S   = Xc^T Xc / (N - 1)                         -> C x C covariance matrix
  S v = lambda v                                   -> eigenvectors v (the axes), eigenvalues lambda (the variance on them)
  sort by lambda, largest first; W = first K eigenvectors as rows (K x C)
  forward  : Z     = Xc W^T          (N x K)  the reduced data
  inverse  : X_hat = Z W + mu        (N x C)  best reconstruction from K numbers per pixel
  explained variance ratio of component i = lambda_i / sum(lambda)
"""
import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components        # how many components to keep after fit (None = all)
        self.mean_ = None                       # (C,)
        self.components_ = None                 # (C_kept, C), one principal axis per row
        self.explained_variance_ = None         # eigenvalues, largest first
        self.explained_variance_ratio_ = None
        self.n_samples_ = None
        self.n_features_ = None

    # ------------------------------------------------------------------ fitting
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        n, c = x.shape
        if n < 2:
            raise ValueError("PCA needs at least 2 samples")
        self.n_samples_, self.n_features_ = n, c

        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_                                   # centred data
        cov = (xc.T @ xc) / (n - 1)                           # C x C covariance matrix

        # eigh is for symmetric matrices: exact, real results, eigenvalues in ascending order
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]                     # largest variance first
        eigvals = np.clip(eigvals[order], 0.0, None)          # tiny negatives are round-off
        eigvecs = eigvecs[:, order].T                         # rows = components

        # An eigenvector's sign is arbitrary (v and -v are both valid). Fix it so the largest
        # absolute weight is positive: the same data then always gives the same model.
        signs = np.sign(eigvecs[np.arange(c), np.abs(eigvecs).argmax(axis=1)])
        signs[signs == 0] = 1
        eigvecs *= signs[:, None]

        k = c if self.n_components is None else int(self.n_components)
        total = eigvals.sum()
        self.components_ = eigvecs[:k]
        self.explained_variance_ = eigvals[:k]
        self.explained_variance_ratio_ = eigvals[:k] / total if total > 0 else np.zeros(k)
        self.all_explained_variance_ratio_ = eigvals / total if total > 0 else np.zeros(c)
        return self

    # ------------------------------------------------------------- projections
    def _w(self, k):
        k = self.components_.shape[0] if k is None else int(k)
        if not 1 <= k <= self.components_.shape[0]:
            raise ValueError(f"k must be between 1 and {self.components_.shape[0]}, got {k}")
        return self.components_[:k]

    def transform(self, x, k=None):
        """C channels -> K components per pixel. Because components are nested, one fitted model
        serves every K: the first K rows of the full model are exactly the K-component model."""
        return (np.asarray(x, dtype=np.float64) - self.mean_) @ self._w(k).T

    def inverse_transform(self, z, k=None):
        """K components -> C channels (the reconstruction)."""
        z = np.asarray(z, dtype=np.float64)
        return z @ self._w(z.shape[1] if k is None else k) + self.mean_

    def components_for_variance(self, threshold):
        """Smallest K whose cumulative explained variance reaches `threshold` (e.g. 0.95)."""
        cum = np.cumsum(self.all_explained_variance_ratio_)
        return int(np.searchsorted(cum, threshold - 1e-12) + 1)

    def n_model_values(self, k):
        """Floats a receiver needs to invert a K-component reduction: mean (C) + projection (K x C)."""
        return self.n_features_ + k * self.n_features_

    # ------------------------------------------------------------------ saving
    def state(self):
        return {"mean": self.mean_, "components": self.components_,
                "explained_variance": self.explained_variance_,
                "explained_variance_ratio": self.explained_variance_ratio_,
                "all_explained_variance_ratio": self.all_explained_variance_ratio_,
                "n_samples": self.n_samples_, "n_features": self.n_features_}

    @classmethod
    def from_state(cls, d):
        p = cls()
        p.mean_ = np.asarray(d["mean"])
        p.components_ = np.asarray(d["components"])
        p.explained_variance_ = np.asarray(d["explained_variance"])
        p.explained_variance_ratio_ = np.asarray(d["explained_variance_ratio"])
        p.all_explained_variance_ratio_ = np.asarray(d["all_explained_variance_ratio"])
        p.n_samples_ = int(d["n_samples"])
        p.n_features_ = int(d["n_features"])
        return p
